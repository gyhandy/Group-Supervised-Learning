"""solver.py"""

import warnings
warnings.filterwarnings("ignore")

import os
from tqdm import tqdm
import visdom

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image

from utils import cuda, grid2gif
from model_share import Generator_fc, Generator_fc_dsprites
from dataset import return_data
from PIL import Image
import torch.nn as nn
import functools
import networks
from torchvision import transforms



class Solver(object):
    def __init__(self, args):
        #GPU
        self.use_cuda = args.cuda and torch.cuda.is_available()

        #dataset and Z-space
        if args.dataset.lower() == 'ilab_20m' or args.dataset.lower() == 'ilab_20m_custom':
            self.nc = 3
            self.z_dim = 100 #'dimension of the latent representation z'
            # id: 0~60; back: 60~80; pose: 80~100
            self.z_pose_dim = 20  # 'dimension of the pose latent representation in z'
            self.z_back_dim = 20  # 'dimension of the background latent representation in z'
            self.z_id_dim = self.z_dim - self.z_pose_dim - self.z_back_dim # 'dimension of the id latent representation in z'

        elif args.dataset.lower() == 'fonts':
            self.nc = 3
            self.z_dim = 100 #'dimension of the latent representation z'
            # content(letter): 0~20; size: 20~40; font_color: 40~60; back_color: 60~80; style(font): 20
            self.z_content_dim = 20  # 'dimension of the z_content (letter) latent representation in z'
            self.z_size_dim = 20  # 'dimension of the z_size latent representation in z'
            self.z_font_color_dim = 20  # 'dimension of the z_font_color latent representation in z'
            self.z_back_color_dim = 20  # 'dimension of the z_back_color latent representation in z'
            self.z_style_dim = 20  # 'dimension of the z_style latent representation in z'

            self.z_content_start_dim = 0
            self.z_size_start_dim = 20
            self.z_font_color_start_dim = 40
            self.z_back_color_start_dim = 60
            self.z_style_start_dim = 80

        elif args.dataset.lower() == 'rafd':
            self.nc = 3
            self.z_dim = 100
            self.z_pose_dim = 20
            self.z_expression_dim = 20
            self.z_id_dim = self.z_dim - self.z_pose_dim - self.z_expression_dim
        elif args.dataset.lower() == 'dsprites':
            self.decoder_dist = 'bernoulli'
            self.nc = 1
            self.z_dim = 10  # 'dimension of the latent representation z'
            self.z_content_dim = 2  # 'dimension of the z_content (letter) latent representation in z'
            self.z_size_dim = 2  # 'dimension of the z_size latent representation in z'
            self.z_font_color_dim = 2 # 'dimension of the z_font_color latent representation in z'
            self.z_back_color_dim = 2  # 'dimension of the z_back_color latent representation in z'
            self.z_style_dim = 2  # 'dimension of the z_style latent representation in z'

            self.z_content_start_dim = 0
            self.z_size_start_dim = 2
            self.z_font_color_start_dim = 4
            self.z_back_color_start_dim = 6
            self.z_style_start_dim = 8
        else:
            raise NotImplementedError
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = args.dataset
        if args.train: # train mode
            self.train = True
        else: # test mode
            self.train = False
            args.batch_size = 1
            self.pretrain_model_path = args.pretrain_model_path
            self.test_img_path = args.test_img_path
        self.batch_size = args.batch_size
        self.data_loader = return_data(args) ### key

        # model training param
        self.g_conv_dim = args.g_conv_dim
        self.g_repeat_num = args.g_repeat_num
        self.norm_layer = get_norm_layer(norm_type=args.norm)
        self.max_iter = args.max_iter
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2

        self.lambda_combine = args.lambda_combine
        self.lambda_unsup = args.lambda_unsup
        if args.dataset.lower() == 'dsprites':
            self.Autoencoder = Generator_fc_dsprites(self.nc, self.g_conv_dim, self.g_repeat_num, self.z_dim)
        else:
            self.Autoencoder = Generator_fc(self.nc, self.g_conv_dim, self.g_repeat_num, self.z_dim)
        self.Autoencoder.to(self.device)
        self.auto_optim = optim.Adam(self.Autoencoder.parameters(), lr=self.lr,
                                     betas=(self.beta1, self.beta2))

        # log and save
        self.log_dir = './checkpoints/' + args.viz_name
        self.model_save_dir = args.model_save_dir
        self.viz_name = args.viz_name
        self.viz_port = args.viz_port
        self.viz_on = args.viz_on
        self.win_recon = None
        self.win_combine_sup = None
        self.win_combine_unsup = None

        self.gather_step = args.gather_step
        self.gather = DataGather()
        self.display_step = args.display_step
        if self.viz_on:
            self.viz = visdom.Visdom(port=self.viz_port)

        self.resume_iters = args.resume_iters
        self.ckpt_dir = os.path.join(args.ckpt_dir, args.viz_name)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)
        self.save_step = args.save_step
        self.save_output = args.save_output
        self.output_dir = os.path.join(args.output_dir, args.viz_name)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

    def restore_model(self, resume_iters):
        """Restore the trained generator"""
        if resume_iters == 'pretrained':
            print('Loading the pretrained models from  {}...'.format(self.pretrain_model_path))
            self.Autoencoder.load_state_dict(torch.load(self.pretrain_model_path, map_location=lambda storage, loc: storage))
            print("=> loaded checkpoint '{} '".format(self.pretrain_model_path))
        else: # not test
            print('Loading the trained models from step {}...'.format(resume_iters))
            Auto_path = os.path.join(self.model_save_dir, self.viz_name, '{}-Auto.ckpt'.format(resume_iters))
            self.Autoencoder.load_state_dict(torch.load(Auto_path, map_location=lambda storage, loc: storage))
            print("=> loaded checkpoint '{} (iter {})'".format(self.viz_name, resume_iters))

    # For ilab20M and ilab20m_custom dataset
    def train_ilab20m(self):
        # self.net_mode(train=True)
        out = False
        # Start training from scratch or resume training.
        self.global_iter = 0
        if self.resume_iters: # if resume the previous training
            self.global_iter = self.resume_iters
            self.restore_model(self.resume_iters)
        pbar = tqdm(total=self.max_iter)
        pbar.update(self.global_iter) # update the current iter if we resume training
        while not out:
            for sup_package in self.data_loader:
                # id, back, pose: 60, 20, 20
                A_img = sup_package['A']
                B_img = sup_package['B']
                C_img = sup_package['C']
                D_img = sup_package['D']
                self.global_iter += 1
                pbar.update(1)

                A_img = Variable(cuda(A_img, self.use_cuda))
                B_img = Variable(cuda(B_img, self.use_cuda))
                C_img = Variable(cuda(C_img, self.use_cuda))
                D_img = Variable(cuda(D_img, self.use_cuda))

                ## 1. A B C seperate(first400: id last600 background)
                A_recon, A_z = self.Autoencoder(A_img)
                B_recon, B_z = self.Autoencoder(B_img)
                C_recon, C_z = self.Autoencoder(C_img)
                D_recon, D_z = self.Autoencoder(D_img)

                A_z_id = A_z[:, 0:self.z_id_dim] # 0-60
                A_z_back = A_z[:, self.z_id_dim:self.z_id_dim + self.z_back_dim] # 60-80
                A_z_pose = A_z[:, self.z_id_dim + self.z_back_dim:] #80-100
                B_z_id = B_z[:, 0:self.z_id_dim]
                B_z_back = B_z[:, self.z_id_dim:self.z_id_dim + self.z_back_dim]
                B_z_pose = B_z[:, self.z_id_dim + self.z_back_dim:]
                C_z_id = C_z[:, 0:self.z_id_dim]
                C_z_back = C_z[:, self.z_id_dim:self.z_id_dim + self.z_back_dim]
                C_z_pose = C_z[:, self.z_id_dim + self.z_back_dim:]
                D_z_id = D_z[:, 0:self.z_id_dim]
                D_z_back = D_z[:, self.z_id_dim:self.z_id_dim + self.z_back_dim]
                D_z_pose = D_z[:, self.z_id_dim + self.z_back_dim:]

                ## 2. combine with strong supervise
                # C A same pose diff id, back
                ApCo_combine_2C = torch.cat((C_z_id, C_z_back), dim=1)
                ApCo_combine_2C = torch.cat((ApCo_combine_2C, A_z_pose), dim=1)
                mid_ApCo = self.Autoencoder.fc_decoder(ApCo_combine_2C)
                mid_ApCo = mid_ApCo.view(ApCo_combine_2C.shape[0], 256, 8, 8)
                ApCo_2C = self.Autoencoder.decoder(mid_ApCo)

                AoCp_combine_2A = torch.cat((A_z_id, A_z_back), dim=1)
                AoCp_combine_2A = torch.cat((AoCp_combine_2A, C_z_pose), dim=1)
                mid_AoCp = self.Autoencoder.fc_decoder(AoCp_combine_2A)
                mid_AoCp = mid_AoCp.view(AoCp_combine_2A.shape[0], 256, 8, 8)
                AoCp_2A = self.Autoencoder.decoder(mid_AoCp)

                # C B same id diff pose, back
                BaCo_combine_2C = torch.cat((B_z_id, C_z_back), dim=1)
                BaCo_combine_2C = torch.cat((BaCo_combine_2C, C_z_pose), dim=1)
                mid_BaCo = self.Autoencoder.fc_decoder(BaCo_combine_2C)
                mid_BaCo = mid_BaCo.view(BaCo_combine_2C.shape[0], 256, 8, 8)
                BaCo_2C = self.Autoencoder.decoder(mid_BaCo)

                BoCa_combine_2B = torch.cat((C_z_id, B_z_back), dim=1)
                BoCa_combine_2B = torch.cat((BoCa_combine_2B, B_z_pose), dim=1)
                mid_BoCa = self.Autoencoder.fc_decoder(BoCa_combine_2B)
                mid_BoCa = mid_BoCa.view(BoCa_combine_2B.shape[0], 256, 8, 8)
                BoCa_2B = self.Autoencoder.decoder(mid_BoCa)

                # C D same background diff id, pose
                DbCo_combine_2C = torch.cat((C_z_id, D_z_back), dim=1)
                DbCo_combine_2C = torch.cat((DbCo_combine_2C, C_z_pose), dim=1)
                mid_DbCo = self.Autoencoder.fc_decoder(DbCo_combine_2C)
                mid_DbCo = mid_DbCo.view(DbCo_combine_2C.shape[0], 256, 8, 8)
                DbCo_2C = self.Autoencoder.decoder(mid_DbCo)

                DoCb_combine_2D = torch.cat((D_z_id, C_z_back), dim=1)
                DoCb_combine_2D = torch.cat((DoCb_combine_2D, D_z_pose), dim=1)
                mid_DoCb = self.Autoencoder.fc_decoder(DoCb_combine_2D)
                mid_DoCb = mid_DoCb.view(DoCb_combine_2D.shape[0], 256, 8, 8)
                DoCb_2D = self.Autoencoder.decoder(mid_DoCb)

                # combine_2C
                ApBaDb_combine_2C = torch.cat((B_z_id, D_z_back), dim=1)
                ApBaDb_combine_2C = torch.cat((ApBaDb_combine_2C, A_z_pose), dim=1)
                mid_ApBaDb = self.Autoencoder.fc_decoder(ApBaDb_combine_2C)
                mid_ApBaDb = mid_ApBaDb.view(ApBaDb_combine_2C.shape[0], 256, 8, 8)
                ApBaDb_2C = self.Autoencoder.decoder(mid_ApBaDb)



                # '''  need unsupervise '''
                AaBpDb_combine_2N = torch.cat((A_z_id, D_z_back), dim=1)
                AaBpDb_combine_2N = torch.cat((AaBpDb_combine_2N, B_z_pose), dim=1)
                mid_AaBpDb = self.Autoencoder.fc_decoder(AaBpDb_combine_2N)
                mid_AaBpDb = mid_AaBpDb.view(AaBpDb_combine_2N.shape[0], 256, 8, 8)
                AaBpDb_2N = self.Autoencoder.decoder(mid_AaBpDb)

                '''
                optimize for autoencoder
                '''
                # 1. recon_loss
                A_recon_loss = torch.mean(torch.abs(A_img - A_recon))
                B_recon_loss = torch.mean(torch.abs(B_img - B_recon))
                C_recon_loss = torch.mean(torch.abs(C_img - C_recon))
                D_recon_loss = torch.mean(torch.abs(D_img - D_recon))
                recon_loss = A_recon_loss + B_recon_loss + C_recon_loss + D_recon_loss

                # 2. sup_combine_loss
                ApCo_2C_loss = torch.mean(torch.abs(C_img - ApCo_2C))
                AoCp_2A_loss = torch.mean(torch.abs(A_img - AoCp_2A))
                BaCo_2C_loss = torch.mean(torch.abs(C_img - BaCo_2C))
                BoCa_2B_loss = torch.mean(torch.abs(B_img - BoCa_2B))
                DbCo_2C_loss = torch.mean(torch.abs(C_img - DbCo_2C))
                DoCb_2D_loss = torch.mean(torch.abs(D_img - DoCb_2D))
                ApBaDb_2C_loss = torch.mean(torch.abs(C_img - ApBaDb_2C))
                combine_sup_loss = ApCo_2C_loss + AoCp_2A_loss + BaCo_2C_loss + BoCa_2B_loss + DbCo_2C_loss + DoCb_2D_loss + ApBaDb_2C_loss

                # 3. unsup_combine_loss
                _, AaBpDb_z = self.Autoencoder(AaBpDb_2N)
                combine_unsup_loss = torch.mean(torch.abs(A_z_id - AaBpDb_z[:, 0:self.z_id_dim])) + \
                                     torch.mean(torch.abs(D_z_back - AaBpDb_z[:, self.z_id_dim:self.z_id_dim + self.z_back_dim])) + \
                                     torch.mean(torch.abs(B_z_pose - AaBpDb_z[:, self.z_id_dim + self.z_back_dim:]))

                # Whole loss
                vae_unsup_loss = recon_loss + self.lambda_combine * combine_sup_loss + self.lambda_unsup * combine_unsup_loss
                self.auto_optim.zero_grad()
                vae_unsup_loss.backward()
                self.auto_optim.step()

                #　save the log
                f = open(self.log_dir + '/log.txt', 'a')
                f.writelines(['\n', '[{}] recon_loss:{:.3f}  combine_sup_loss:{:.3f}  combine_unsup_loss:{:.3f}'.format(
                        self.global_iter, recon_loss.data, combine_sup_loss.data, combine_unsup_loss.data)])
                f.close()


                if self.viz_on and self.global_iter%self.gather_step == 0: # gather loss information
                    self.gather.insert(iter=self.global_iter,recon_loss=recon_loss.data,
                                    combine_sup_loss=combine_sup_loss.data, combine_unsup_loss=combine_unsup_loss.data)

                if self.global_iter%self.display_step == 0:
                    pbar.write('[{}] recon_loss:{:.3f}  combine_sup_loss:{:.3f}  combine_unsup_loss:{:.3f}'.format(
                        self.global_iter, recon_loss.data, combine_sup_loss.data, combine_unsup_loss.data))

                    if self.viz_on:
                        self.gather.insert(images=A_img.data)
                        self.gather.insert(images=B_img.data)
                        self.gather.insert(images=C_img.data)
                        self.gather.insert(images=D_img.data)
                        self.gather.insert(images=F.sigmoid(A_recon).data)
                        self.ilab20m_viz_reconstruction()
                        self.viz_lines()
                        '''
                        combine show
                        '''
                        self.gather.insert(combine_supimages=F.sigmoid(AoCp_2A).data)
                        self.gather.insert(combine_supimages=F.sigmoid(BoCa_2B).data)
                        self.gather.insert(combine_supimages=F.sigmoid(DbCo_2C).data)
                        self.gather.insert(combine_supimages=F.sigmoid(DoCb_2D).data)
                        self.ilab20m_viz_combine_recon()

                        self.gather.insert(combine_unsupimages=F.sigmoid(ApBaDb_2C).data)
                        self.gather.insert(combine_unsupimages=F.sigmoid(AaBpDb_2N).data)
                        self.ilab20m_viz_combine_unsuprecon()
                        self.gather.flush()
                # Save model checkpoints.
                if self.global_iter%self.save_step == 0:
                    Auto_path = os.path.join(self.model_save_dir, self.viz_name, '{}-Auto.ckpt'.format(self.global_iter))
                    torch.save(self.Autoencoder.state_dict(), Auto_path)
                    print('Saved model checkpoints into {}/{}...'.format(self.model_save_dir, self.viz_name))

                if self.global_iter >= self.max_iter:
                    out = True
                    break

        pbar.write("[Training Finished]")
        pbar.close()
    def train_ilab20m_custom(self):
        # self.net_mode(train=True)
        out = False
        # Start training from scratch or resume training.
        self.global_iter = 0
        if self.resume_iters:
            self.global_iter = self.resume_iters
            self.restore_model(self.resume_iters)

        pbar = tqdm(total=self.max_iter)
        pbar.update(self.global_iter)
        while not out:
            for sup_package in self.data_loader:
                if self.global_iter == 121:
                    print(121)
                # appe, pose, combine
                A_img = sup_package['A']
                B_img = sup_package['B']
                C_img = sup_package['C']
                D_img = sup_package['D']
                C_img1 = sup_package['C1']
                C_img2 = sup_package['C2']
                self.global_iter += 1
                pbar.update(1)

                A_img = Variable(cuda(A_img, self.use_cuda))
                B_img = Variable(cuda(B_img, self.use_cuda))
                C_img = Variable(cuda(C_img, self.use_cuda))
                D_img = Variable(cuda(D_img, self.use_cuda))
                C_img1 = Variable(cuda(C_img1, self.use_cuda))
                C_img2 = Variable(cuda(C_img2, self.use_cuda))

                ## 1. A B C seperate(first400: id last600 background)
                A_recon, A_z = self.Autoencoder(A_img)
                B_recon, B_z = self.Autoencoder(B_img)
                C_recon, C_z = self.Autoencoder(C_img)
                D_recon, D_z = self.Autoencoder(D_img)

                C_recon1, C_z1 = self.Autoencoder(C_img1)
                C_recon2, C_z2 = self.Autoencoder(C_img2)

                A_z_appe = A_z[:, 0:self.z_id_dim] # 0-700
                A_z_back = A_z[:, self.z_id_dim:self.z_id_dim + self.z_back_dim] # 700-800
                A_z_pose = A_z[:, self.z_id_dim + self.z_back_dim:] #800-1000
                B_z_appe = B_z[:, 0:self.z_id_dim]
                B_z_back = B_z[:, self.z_id_dim:self.z_id_dim + self.z_back_dim]
                B_z_pose = B_z[:, self.z_id_dim + self.z_back_dim:]
                C_z_appe = C_z[:, 0:self.z_id_dim]
                C_z_back = C_z[:, self.z_id_dim:self.z_id_dim + self.z_back_dim]
                C_z_pose = C_z[:, self.z_id_dim + self.z_back_dim:]
                D_z_appe = D_z[:, 0:self.z_id_dim]
                D_z_back = D_z[:, self.z_id_dim:self.z_id_dim + self.z_back_dim]
                D_z_pose = D_z[:, self.z_id_dim + self.z_back_dim:]

                C_z_appe1 = C_z1[:, 0:self.z_id_dim]
                C_z_back1 = C_z1[:, self.z_id_dim:self.z_id_dim + self.z_back_dim]
                C_z_pose1 = C_z1[:, self.z_id_dim + self.z_back_dim:]

                C_z_appe2 = C_z2[:, 0:self.z_id_dim]
                C_z_back2 = C_z2[:, self.z_id_dim:self.z_id_dim + self.z_back_dim]
                C_z_pose2 = C_z2[:, self.z_id_dim + self.z_back_dim:]

                ## 2. combine with strong supervise
                # C A same pose diff id, back
                ApCo_combine_2C = torch.cat((C_z_appe1, C_z_back1), dim=1)
                ApCo_combine_2C = torch.cat((ApCo_combine_2C, A_z_pose), dim=1)
                mid_ApCo = self.Autoencoder.fc_decoder(ApCo_combine_2C)
                mid_ApCo = mid_ApCo.view(ApCo_combine_2C.shape[0], 256, 8, 8)
                ApCo_2C = self.Autoencoder.decoder(mid_ApCo)

                AoCp_combine_2A = torch.cat((A_z_appe, A_z_back), dim=1)
                AoCp_combine_2A = torch.cat((AoCp_combine_2A, C_z_pose1), dim=1)
                mid_AoCp = self.Autoencoder.fc_decoder(AoCp_combine_2A)
                mid_AoCp = mid_AoCp.view(AoCp_combine_2A.shape[0], 256, 8, 8)
                AoCp_2A = self.Autoencoder.decoder(mid_AoCp)

                # C B same id diff pose, back
                BaCo_combine_2C = torch.cat((B_z_appe, C_z_back), dim=1)
                BaCo_combine_2C = torch.cat((BaCo_combine_2C, C_z_pose), dim=1)
                mid_BaCo = self.Autoencoder.fc_decoder(BaCo_combine_2C)
                mid_BaCo = mid_BaCo.view(BaCo_combine_2C.shape[0], 256, 8, 8)
                BaCo_2C = self.Autoencoder.decoder(mid_BaCo)

                BoCa_combine_2B = torch.cat((C_z_appe, B_z_back), dim=1)
                BoCa_combine_2B = torch.cat((BoCa_combine_2B, B_z_pose), dim=1)
                mid_BoCa = self.Autoencoder.fc_decoder(BoCa_combine_2B)
                mid_BoCa = mid_BoCa.view(BoCa_combine_2B.shape[0], 256, 8, 8)
                BoCa_2B = self.Autoencoder.decoder(mid_BoCa)

                # C D same background diff id, pose
                DbCo_combine_2C = torch.cat((C_z_appe2, D_z_back), dim=1)
                DbCo_combine_2C = torch.cat((DbCo_combine_2C, C_z_pose2), dim=1)
                mid_DbCo = self.Autoencoder.fc_decoder(DbCo_combine_2C)
                mid_DbCo = mid_DbCo.view(DbCo_combine_2C.shape[0], 256, 8, 8)
                DbCo_2C = self.Autoencoder.decoder(mid_DbCo)

                DoCb_combine_2D = torch.cat((D_z_appe, C_z_back2), dim=1)
                DoCb_combine_2D = torch.cat((DoCb_combine_2D, D_z_pose), dim=1)
                mid_DoCb = self.Autoencoder.fc_decoder(DoCb_combine_2D)
                mid_DoCb = mid_DoCb.view(DoCb_combine_2D.shape[0], 256, 8, 8)
                DoCb_2D = self.Autoencoder.decoder(mid_DoCb)

                # combine_2C
                ApBaDb_combine_2C = torch.cat((B_z_appe, D_z_back), dim=1)
                ApBaDb_combine_2C = torch.cat((ApBaDb_combine_2C, A_z_pose), dim=1)
                mid_ApBaDb = self.Autoencoder.fc_decoder(ApBaDb_combine_2C)
                mid_ApBaDb = mid_ApBaDb.view(ApBaDb_combine_2C.shape[0], 256, 8, 8)
                ApBaDb_2C = self.Autoencoder.decoder(mid_ApBaDb)



                # '''  need unsupervise '''
                AaBpDb_combine_2N = torch.cat((A_z_appe, D_z_back), dim=1)
                AaBpDb_combine_2N = torch.cat((AaBpDb_combine_2N, B_z_pose), dim=1)
                mid_AaBpDb = self.Autoencoder.fc_decoder(AaBpDb_combine_2N)
                mid_AaBpDb = mid_AaBpDb.view(AaBpDb_combine_2N.shape[0], 256, 8, 8)
                AaBpDb_2N = self.Autoencoder.decoder(mid_AaBpDb)

                # '''  need unsupervise '''
                # AaBp_combine_2N = torch.cat((A_z_appe, C_z_back), dim=1)
                # AaBp_combine_2N = torch.cat((AaBp_combine_2N, B_z_pose), dim=1)
                # mid_AaBp = self.Autoencoder.fc_decoder(AaBp_combine_2N)
                # mid_AaBp = mid_AaBp.view(AaBp_combine_2N.shape[0], 256, 8, 8)
                # AaBp_2N = self.Autoencoder.decoder(mid_AaBp)


                '''
                optimize for autoencoder
                '''

                # 1. recon_loss
                A_recon_loss = torch.mean(torch.abs(A_img - A_recon))
                B_recon_loss = torch.mean(torch.abs(B_img - B_recon))
                C_recon_loss = torch.mean(torch.abs(C_img - C_recon))
                D_recon_loss = torch.mean(torch.abs(D_img - D_recon))
                recon_loss = A_recon_loss + B_recon_loss + C_recon_loss + D_recon_loss

                # 2. sup_combine_loss
                ApCo_2C_loss = torch.mean(torch.abs(C_img1 - ApCo_2C))
                AoCp_2A_loss = torch.mean(torch.abs(A_img - AoCp_2A))
                BaCo_2C_loss = torch.mean(torch.abs(C_img - BaCo_2C))
                BoCa_2B_loss = torch.mean(torch.abs(B_img - BoCa_2B))
                DbCo_2C_loss = torch.mean(torch.abs(C_img2 - DbCo_2C))
                DoCb_2D_loss = torch.mean(torch.abs(D_img - DoCb_2D))
                ApBaDb_2C_loss = torch.mean(torch.abs(C_img - ApBaDb_2C))
                combine_sup_loss = ApCo_2C_loss + AoCp_2A_loss + BaCo_2C_loss + BoCa_2B_loss + DbCo_2C_loss + DoCb_2D_loss #+ ApBaDb_2C_loss

                # 3. unsup_combine_loss
                _, AaBpDb_z = self.Autoencoder(AaBpDb_2N)
                combine_unsup_loss = torch.mean(torch.abs(A_z_appe - AaBpDb_z[:, 0:self.z_id_dim])) + torch.mean(torch.abs(D_z_back - AaBpDb_z[:, self.z_id_dim:self.z_id_dim + self.z_back_dim])) + torch.mean(torch.abs(B_z_pose - AaBpDb_z[:, self.z_id_dim + self.z_back_dim:]))

                # whole loss
                vae_unsup_loss = recon_loss + self.lambda_combine * combine_sup_loss + self.lambda_unsup * combine_unsup_loss
                self.auto_optim.zero_grad()
                vae_unsup_loss.backward()
                self.auto_optim.step()

                #　save the log
                f = open(self.log_dir + '/log.txt', 'a')
                f.writelines(['\n', '[{}] recon_loss:{:.3f}  combine_sup_loss:{:.3f}  combine_unsup_loss:{:.3f}'.format(
                        self.global_iter, recon_loss.data, combine_sup_loss.data, combine_unsup_loss.data)])
                f.close()
                print(['\n', '[{}] recon_loss:{:.3f}  combine_sup_loss:{:.3f}  combine_unsup_loss:{:.3f}'.format(
                        self.global_iter, recon_loss.data, combine_sup_loss.data, combine_unsup_loss.data)])


                if self.viz_on and self.global_iter%self.gather_step == 0:
                    self.gather.insert(iter=self.global_iter,recon_loss=recon_loss.data,
                                    combine_sup_loss=combine_sup_loss.data, combine_unsup_loss=combine_unsup_loss.data)

                if self.global_iter%self.display_step == 0:
                    pbar.write('[{}] recon_loss:{:.3f}  combine_sup_loss:{:.3f}  combine_unsup_loss:{:.3f}'.format(
                        self.global_iter, recon_loss.data, combine_sup_loss.data, combine_unsup_loss.data))

                    if self.viz_on:
                        self.gather.insert(images=A_img.data)
                        self.gather.insert(images=B_img.data)
                        self.gather.insert(images=C_img.data)
                        self.gather.insert(images=D_img.data)
                        self.gather.insert(images=F.sigmoid(A_recon).data)
                        self.ilab20m_viz_reconstruction()
                        self.viz_lines()
                        '''
                        combine show
                        '''
                        self.gather.insert(combine_supimages=F.sigmoid(AoCp_2A).data)
                        self.gather.insert(combine_supimages=F.sigmoid(BoCa_2B).data)
                        self.gather.insert(combine_supimages=F.sigmoid(DbCo_2C).data)
                        self.gather.insert(combine_supimages=F.sigmoid(DoCb_2D).data)
                        self.ilab20m_viz_combine_recon()

                        self.gather.insert(combine_unsupimages=F.sigmoid(ApBaDb_2C).data)
                        self.gather.insert(combine_unsupimages=F.sigmoid(AaBpDb_2N).data)
                        self.ilab20m_viz_combine_unsuprecon()
                        # self.viz_combine(x)
                        self.gather.flush()
                # Save model checkpoints.
                if self.global_iter%self.save_step == 0:
                    Auto_path = os.path.join(self.model_save_dir, self.viz_name, '{}-Auto.ckpt'.format(self.global_iter))
                    torch.save(self.Autoencoder.state_dict(), Auto_path)
                    print('Saved model checkpoints into {}/{}...'.format(self.model_save_dir, self.viz_name))


                if self.global_iter >= self.max_iter:
                    out = True
                    break

        pbar.write("[Training Finished]")
        pbar.close()
    def test_ilab20m(self):
        # self.net_mode(train=False)
        # load pretrained model
        self.restore_model('pretrained')
        for index, sup_package in enumerate(self.data_loader):
            # id, back, pose
            A_img = sup_package['A']
            B_img = sup_package['B']
            D_img = sup_package['D']

            A_img = Variable(cuda(A_img, self.use_cuda))
            B_img = Variable(cuda(B_img, self.use_cuda))
            D_img = Variable(cuda(D_img, self.use_cuda))

            ## 1. get latent
            A_recon, A_z = self.Autoencoder(A_img)
            B_recon, B_z = self.Autoencoder(B_img)
            D_recon, D_z = self.Autoencoder(D_img)

            A_z_id = A_z[:, 0:self.z_id_dim] # 0-60
            D_z_back = D_z[:, self.z_id_dim:self.z_id_dim + self.z_back_dim]  # 60-80
            B_z_pose = B_z[:, self.z_id_dim + self.z_back_dim:] # 80-100


            ## 2. combine for target
            AaBpDb_combine_2N = torch.cat((A_z_id, D_z_back, B_z_pose), dim=1)
            # AaBpDb_combine_2N = torch.cat((AaBpDb_combine_2N, B_z_pose), dim=1)
            mid_AaBpDb = self.Autoencoder.fc_decoder(AaBpDb_combine_2N)
            mid_AaBpDb = mid_AaBpDb.view(AaBpDb_combine_2N.shape[0], 256, 8, 8)
            AaBpDb_2N = self.Autoencoder.decoder(mid_AaBpDb)

            # save synthesized image
            self.test_iter = index
            self.gather.insert(test=F.sigmoid(AaBpDb_2N).data)
            self.ilab20m_viz_test()
            self.gather.flush()

    def ilab20m_save_sample_img(self, tensor, mode):
        unloader = transforms.ToPILImage()
        dir = os.path.join(self.model_save_dir, self.viz_name, 'sample_img')
        if not os.path.exists(dir):
            os.makedirs(dir)
        image = tensor.cpu().clone()  # we clone the tensor to not do changes on it

        if mode == 'recon':
            image_ori_A = image[0].squeeze(0)  # remove the fake batch dimension
            image_ori_B = image[1].squeeze(0)
            image_ori_C = image[2].squeeze(0)
            image_ori_D = image[3].squeeze(0)
            image_recon = image[4].squeeze(0)

            image_ori_A = unloader(image_ori_A)
            image_ori_B = unloader(image_ori_B)
            image_ori_C = unloader(image_ori_C)
            image_ori_D = unloader(image_ori_D)
            image_recon = unloader(image_recon)

            image_ori_A.save(os.path.join(dir, '{}-A_img.png'.format(self.global_iter)))
            image_ori_B.save(os.path.join(dir, '{}-B_img.png'.format(self.global_iter)))
            image_ori_C.save(os.path.join(dir, '{}-C_img.png'.format(self.global_iter)))
            image_ori_D.save(os.path.join(dir, '{}-D_img.png'.format(self.global_iter)))
            image_recon.save(os.path.join(dir, '{}-A_img_recon.png'.format(self.global_iter)))
        elif mode == 'combine_sup':
            image_AoCp_2A = image[0].squeeze(0)  # remove the fake batch dimension
            image_BoCa_2B = image[1].squeeze(0)
            image_DbCo_2C = image[2].squeeze(0)
            image_DoCb_2D = image[3].squeeze(0)

            image_AoCp_2A = unloader(image_AoCp_2A)
            image_BoCa_2B = unloader(image_BoCa_2B)
            image_DbCo_2C = unloader(image_DbCo_2C)
            image_DoCb_2D = unloader(image_DoCb_2D)

            image_AoCp_2A.save(os.path.join(dir, '{}-AoCp_2A.png'.format(self.global_iter)))
            image_BoCa_2B.save(os.path.join(dir, '{}-BoCa_2B.png'.format(self.global_iter)))
            image_DbCo_2C.save(os.path.join(dir, '{}-DbCo_2C.png'.format(self.global_iter)))
            image_DoCb_2D.save(os.path.join(dir, '{}-DoCb_2D.png'.format(self.global_iter)))
        elif mode == 'combine_unsup':
            image_ApBaDb_2C = image[0].squeeze(0)  # remove the fake batch dimension
            image_AaBpDb_2N = image[1].squeeze(0)

            image_ApBaDb_2C = unloader(image_ApBaDb_2C)
            image_AaBpDb_2N = unloader(image_AaBpDb_2N)

            image_ApBaDb_2C.save(os.path.join(dir, '{}-ApBaDb_2C.png'.format(self.global_iter)))
            image_AaBpDb_2N.save(os.path.join(dir, '{}-AaBpDb_2N.png'.format(self.global_iter)))
        elif mode == 'test':
            image_AaBpDb_2N = image
            image_AaBpDb_2N = unloader(image_AaBpDb_2N)
            image_AaBpDb_2N.save(os.path.join(self.output_dir, 'group{}-AaBpDb_2N.png'.format(self.test_iter)))
    def ilab20m_viz_reconstruction(self):
        # self.net_mode(train=False)
        x_A = self.gather.data['images'][0][:100]
        x_A = make_grid(x_A, normalize=True)
        x_B = self.gather.data['images'][1][:100]
        x_B = make_grid(x_B, normalize=True)
        x_C = self.gather.data['images'][2][:100]
        x_C = make_grid(x_C, normalize=True)
        x_D = self.gather.data['images'][3][:100]
        x_D = make_grid(x_D, normalize=True)
        x_A_recon = self.gather.data['images'][4][:100]
        x_A_recon = make_grid(x_A_recon, normalize=True)
        images = torch.stack([x_A, x_B, x_C, x_D, x_A_recon], dim=0).cpu()
        self.viz.images(images, env=self.viz_name+'_reconstruction',
                        opts=dict(title=str(self.global_iter)), nrow=10)
        self.ilab20m_save_sample_img(images, 'recon')
        # self.net_mode(train=True)
    def ilab20m_viz_combine_recon(self):
        # self.net_mode(train=False)
        AoCp_2A = self.gather.data['combine_supimages'][0][:100]
        AoCp_2A = make_grid(AoCp_2A, normalize=True)
        BoCa_2B = self.gather.data['combine_supimages'][1][:100]
        BoCa_2B = make_grid(BoCa_2B, normalize=True)
        DbCo_2C = self.gather.data['combine_supimages'][2][:100]
        DbCo_2C = make_grid(DbCo_2C, normalize=True)
        DoCb_2D = self.gather.data['combine_supimages'][3][:100]
        DoCb_2D = make_grid(DoCb_2D, normalize=True)
        images = torch.stack([AoCp_2A, BoCa_2B, DbCo_2C, DoCb_2D], dim=0).cpu()
        self.viz.images(images, env=self.viz_name+'combine_supimages',
                        opts=dict(title=str(self.global_iter)), nrow=10)
        self.ilab20m_save_sample_img(images, 'combine_sup')
    def ilab20m_viz_combine_unsuprecon(self):
        # self.net_mode(train=False)
        ApBaDb_2C = self.gather.data['combine_unsupimages'][0][:100]
        ApBaDb_2C = make_grid(ApBaDb_2C, normalize=True)
        AaBpDb_2N = self.gather.data['combine_unsupimages'][1][:100]
        AaBpDb_2N = make_grid(AaBpDb_2N, normalize=True)
        images = torch.stack([ApBaDb_2C, AaBpDb_2N], dim=0).cpu()
        self.viz.images(images, env=self.viz_name+'combine_unsupimages',
                        opts=dict(title=str(self.global_iter)), nrow=10)
        self.ilab20m_save_sample_img(images, 'combine_unsup')
    def ilab20m_viz_test(self):
        # self.net_mode(train=False)
        AaBpDb_2N = self.gather.data['test'][0][:100]
        AaBpDb_2N = make_grid(AaBpDb_2N, normalize=True)
        images = AaBpDb_2N
        self.ilab20m_save_sample_img(images, 'test')


    # For Fonts dataset
    def train_fonts(self):
        # self.net_mode(train=True)
        out = False
        # Start training from scratch or resume training.
        self.global_iter = 0
        if self.resume_iters:
            self.global_iter = self.resume_iters
            self.restore_model(self.resume_iters)

        pbar = tqdm(total=self.max_iter)
        pbar.update(self.global_iter)
        while not out:
            for sup_package in self.data_loader:

                A_img = sup_package['A']
                B_img = sup_package['B']
                C_img = sup_package['C']
                D_img = sup_package['D']
                E_img = sup_package['E']
                F_img = sup_package['F']
                self.global_iter += 1
                pbar.update(1)

                A_img = Variable(cuda(A_img, self.use_cuda))
                B_img = Variable(cuda(B_img, self.use_cuda))
                C_img = Variable(cuda(C_img, self.use_cuda))
                D_img = Variable(cuda(D_img, self.use_cuda))
                E_img = Variable(cuda(E_img, self.use_cuda))
                F_img = Variable(cuda(F_img, self.use_cuda))

                ## 1. A B C seperate(first400: id last600 background)
                A_recon, A_z = self.Autoencoder(A_img)
                B_recon, B_z = self.Autoencoder(B_img)
                C_recon, C_z = self.Autoencoder(C_img)
                D_recon, D_z = self.Autoencoder(D_img)
                E_recon, E_z = self.Autoencoder(E_img)
                F_recon, F_z = self.Autoencoder(F_img)
                ''' refer 1: content, 2: size, 3: font-color, 4 back_color, 5 style'''

                A_z_1 = A_z[:, 0:self.z_size_start_dim]  # 0-20
                A_z_2 = A_z[:, self.z_size_start_dim: self.z_font_color_start_dim]  # 20-40
                A_z_3 = A_z[:, self.z_font_color_start_dim: self.z_back_color_start_dim]  # 40-60
                A_z_4 = A_z[:, self.z_back_color_start_dim: self.z_style_start_dim]  # 60-80
                A_z_5 = A_z[:, self.z_style_start_dim:]  # 80-100
                B_z_1 = B_z[:, 0:self.z_size_start_dim]  # 0-20
                B_z_2 = B_z[:, self.z_size_start_dim: self.z_font_color_start_dim]  # 20-40
                B_z_3 = B_z[:, self.z_font_color_start_dim: self.z_back_color_start_dim]  # 40-60
                B_z_4 = B_z[:, self.z_back_color_start_dim: self.z_style_start_dim]  # 60-80
                B_z_5 = B_z[:, self.z_style_start_dim:]  # 80-100
                C_z_1 = C_z[:, 0:self.z_size_start_dim]  # 0-20
                C_z_2 = C_z[:, self.z_size_start_dim: self.z_font_color_start_dim]  # 20-40
                C_z_3 = C_z[:, self.z_font_color_start_dim: self.z_back_color_start_dim]  # 40-60
                C_z_4 = C_z[:, self.z_back_color_start_dim: self.z_style_start_dim]  # 60-80
                C_z_5 = C_z[:, self.z_style_start_dim:]  # 80-100
                D_z_1 = D_z[:, 0:self.z_size_start_dim]  # 0-20
                D_z_2 = D_z[:, self.z_size_start_dim: self.z_font_color_start_dim]  # 20-40
                D_z_3 = D_z[:, self.z_font_color_start_dim: self.z_back_color_start_dim]  # 40-60
                D_z_4 = D_z[:, self.z_back_color_start_dim: self.z_style_start_dim]  # 60-80
                D_z_5 = D_z[:, self.z_style_start_dim:]  # 80-100
                E_z_1 = E_z[:, 0:self.z_size_start_dim]  # 0-20
                E_z_2 = E_z[:, self.z_size_start_dim: self.z_font_color_start_dim]  # 20-40
                E_z_3 = E_z[:, self.z_font_color_start_dim: self.z_back_color_start_dim]  # 40-60
                E_z_4 = E_z[:, self.z_back_color_start_dim: self.z_style_start_dim]  # 60-80
                E_z_5 = E_z[:, self.z_style_start_dim:]  # 80-100
                F_z_1 = F_z[:, 0:self.z_size_start_dim]  # 0-20
                F_z_2 = F_z[:, self.z_size_start_dim: self.z_font_color_start_dim]  # 20-40
                F_z_3 = F_z[:, self.z_font_color_start_dim: self.z_back_color_start_dim]  # 40-60
                F_z_4 = F_z[:, self.z_back_color_start_dim: self.z_style_start_dim]  # 60-80
                F_z_5 = F_z[:, self.z_style_start_dim:]  # 80-100

                ## 2. combine with strong supervise
                ''' refer 1: content, 2: size, 3: font-color, 4 back_color, 5 style'''
                # C A same content 1
                A1Co_combine_2C = torch.cat((A_z_1, C_z_2, C_z_3, C_z_4, C_z_5), dim=1)
                mid_A1Co = self.Autoencoder.fc_decoder(A1Co_combine_2C)
                mid_A1Co = mid_A1Co.view(A1Co_combine_2C.shape[0], 256, 8, 8)
                A1Co_2C = self.Autoencoder.decoder(mid_A1Co)

                AoC1_combine_2A = torch.cat((C_z_1, A_z_2, A_z_3, A_z_4, A_z_5), dim=1)
                mid_AoC1 = self.Autoencoder.fc_decoder(AoC1_combine_2A)
                mid_AoC1 = mid_AoC1.view(AoC1_combine_2A.shape[0], 256, 8, 8)
                AoC1_2A = self.Autoencoder.decoder(mid_AoC1)

                # C B same size 2
                B2Co_combine_2C = torch.cat((C_z_1, B_z_2, C_z_3, C_z_4, C_z_5), dim=1)
                mid_B2Co = self.Autoencoder.fc_decoder(B2Co_combine_2C)
                mid_B2Co = mid_B2Co.view(B2Co_combine_2C.shape[0], 256, 8, 8)
                B2Co_2C = self.Autoencoder.decoder(mid_B2Co)

                BoC2_combine_2B = torch.cat((B_z_1, C_z_2, B_z_3, B_z_4, B_z_5), dim=1)
                mid_BoC2 = self.Autoencoder.fc_decoder(BoC2_combine_2B)
                mid_BoC2 = mid_BoC2.view(BoC2_combine_2B.shape[0], 256, 8, 8)
                BoC2_2B = self.Autoencoder.decoder(mid_BoC2)

                # C D same font_color 3
                D3Co_combine_2C = torch.cat((C_z_1, C_z_2, D_z_3, C_z_4, C_z_5), dim=1)
                mid_D3Co = self.Autoencoder.fc_decoder(D3Co_combine_2C)
                mid_D3Co = mid_D3Co.view(D3Co_combine_2C.shape[0], 256, 8, 8)
                D3Co_2C = self.Autoencoder.decoder(mid_D3Co)

                DoC3_combine_2D = torch.cat((D_z_1, D_z_2, C_z_3, D_z_4, D_z_5), dim=1)
                mid_DoC3 = self.Autoencoder.fc_decoder(DoC3_combine_2D)
                mid_DoC3 = mid_DoC3.view(DoC3_combine_2D.shape[0], 256, 8, 8)
                DoC3_2D = self.Autoencoder.decoder(mid_DoC3)

                # C E same back_color 4
                E4Co_combine_2C = torch.cat((C_z_1, C_z_2, C_z_3, E_z_4, C_z_5), dim=1)
                mid_E4Co = self.Autoencoder.fc_decoder(E4Co_combine_2C)
                mid_E4Co = mid_E4Co.view(E4Co_combine_2C.shape[0], 256, 8, 8)
                E4Co_2C = self.Autoencoder.decoder(mid_E4Co)

                EoC4_combine_2E = torch.cat((E_z_1, E_z_2, E_z_3, C_z_4, E_z_5), dim=1)
                mid_EoC4 = self.Autoencoder.fc_decoder(EoC4_combine_2E)
                mid_EoC4 = mid_EoC4.view(EoC4_combine_2E.shape[0], 256, 8, 8)
                EoC4_2E = self.Autoencoder.decoder(mid_EoC4)

                # C F same style 5
                F5Co_combine_2C = torch.cat((C_z_1, C_z_2, C_z_3, C_z_4, F_z_5), dim=1)
                mid_F5Co = self.Autoencoder.fc_decoder(F5Co_combine_2C)
                mid_F5Co = mid_F5Co.view(F5Co_combine_2C.shape[0], 256, 8, 8)
                F5Co_2C = self.Autoencoder.decoder(mid_F5Co)

                FoC5_combine_2F = torch.cat((F_z_1, F_z_2, F_z_3, F_z_4, C_z_5), dim=1)
                mid_FoC5 = self.Autoencoder.fc_decoder(FoC5_combine_2F)
                mid_FoC5 = mid_FoC5.view(FoC5_combine_2F.shape[0], 256, 8, 8)
                FoC5_2F = self.Autoencoder.decoder(mid_FoC5)

                # combine_2C
                A1B2D3E4F5_combine_2C = torch.cat((A_z_1, B_z_2, D_z_3, E_z_4, F_z_5), dim=1)
                mid_A1B2D3E4F5 = self.Autoencoder.fc_decoder(A1B2D3E4F5_combine_2C)
                mid_A1B2D3E4F5 = mid_A1B2D3E4F5.view(A1B2D3E4F5_combine_2C.shape[0], 256, 8, 8)
                A1B2D3E4F5_2C = self.Autoencoder.decoder(mid_A1B2D3E4F5)

                # '''  need unsupervise '''
                A2B3D4E5F1_combine_2N = torch.cat((F_z_1, A_z_2, B_z_3, D_z_4, E_z_5), dim=1)
                mid_A2B3D4E5F1 = self.Autoencoder.fc_decoder(A2B3D4E5F1_combine_2N)
                mid_A2B3D4E5F1 = mid_A2B3D4E5F1.view(A2B3D4E5F1_combine_2N.shape[0], 256, 8, 8)
                A2B3D4E5F1_2N = self.Autoencoder.decoder(mid_A2B3D4E5F1)

                '''
                optimize for autoencoder
                '''
                # 1. recon_loss
                A_recon_loss = torch.mean(torch.abs(A_img - A_recon))
                B_recon_loss = torch.mean(torch.abs(B_img - B_recon))
                C_recon_loss = torch.mean(torch.abs(C_img - C_recon))
                D_recon_loss = torch.mean(torch.abs(D_img - D_recon))
                E_recon_loss = torch.mean(torch.abs(E_img - E_recon))
                F_recon_loss = torch.mean(torch.abs(F_img - F_recon))
                recon_loss = A_recon_loss + B_recon_loss + C_recon_loss + D_recon_loss + E_recon_loss + F_recon_loss

                # 2. sup_combine_loss
                A1Co_2C_loss = torch.mean(torch.abs(C_img - A1Co_2C))
                AoC1_2A_loss = torch.mean(torch.abs(A_img - AoC1_2A))
                B2Co_2C_loss = torch.mean(torch.abs(C_img - B2Co_2C))
                BoC2_2B_loss = torch.mean(torch.abs(B_img - BoC2_2B))
                D3Co_2C_loss = torch.mean(torch.abs(C_img - D3Co_2C))
                DoC3_2D_loss = torch.mean(torch.abs(D_img - DoC3_2D))
                E4Co_2C_loss = torch.mean(torch.abs(C_img - E4Co_2C))
                EoC4_2E_loss = torch.mean(torch.abs(E_img - EoC4_2E))
                F5Co_2C_loss = torch.mean(torch.abs(C_img - F5Co_2C))
                FoC5_2F_loss = torch.mean(torch.abs(F_img - FoC5_2F))
                A1B2D3E4F5_2C_loss = torch.mean(torch.abs(C_img - A1B2D3E4F5_2C))
                combine_sup_loss = A1Co_2C_loss + AoC1_2A_loss + B2Co_2C_loss + BoC2_2B_loss + D3Co_2C_loss + DoC3_2D_loss + E4Co_2C_loss + EoC4_2E_loss + F5Co_2C_loss + FoC5_2F_loss + A1B2D3E4F5_2C_loss

                # 3. unsup_combine_loss
                _, A2B3D4E5F1_z = self.Autoencoder(A2B3D4E5F1_2N)
                combine_unsup_loss = torch.mean(
                    torch.abs(F_z_1 - A2B3D4E5F1_z[:, 0:self.z_size_start_dim])) + torch.mean(
                    torch.abs(A_z_2 - A2B3D4E5F1_z[:, self.z_size_start_dim: self.z_font_color_start_dim])) \
                                     + torch.mean(
                    torch.abs(B_z_3 - A2B3D4E5F1_z[:, self.z_font_color_start_dim: self.z_back_color_start_dim])) \
                                     + torch.mean(
                    torch.abs(D_z_4 - A2B3D4E5F1_z[:, self.z_back_color_start_dim: self.z_style_start_dim])) \
                                     + torch.mean(torch.abs(E_z_5 - A2B3D4E5F1_z[:, self.z_style_start_dim:]))

                # whole loss
                vae_unsup_loss = recon_loss + self.lambda_combine * combine_sup_loss + self.lambda_unsup * combine_unsup_loss
                self.auto_optim.zero_grad()
                vae_unsup_loss.backward()
                self.auto_optim.step()

                # 　save the log
                f = open(self.log_dir + '/log.txt', 'a')
                f.writelines(['\n', '[{}] recon_loss:{:.3f}  combine_sup_loss:{:.3f}  combine_unsup_loss:{:.3f}'.format(
                    self.global_iter, recon_loss.data, combine_sup_loss.data, combine_unsup_loss.data)])
                f.close()

                if self.viz_on and self.global_iter % self.gather_step == 0:
                    self.gather.insert(iter=self.global_iter, recon_loss=recon_loss.data,
                                       combine_sup_loss=combine_sup_loss.data,
                                       combine_unsup_loss=combine_unsup_loss.data)

                if self.global_iter % self.display_step == 0:
                    pbar.write('[{}] recon_loss:{:.3f}  combine_sup_loss:{:.3f}  combine_unsup_loss:{:.3f}'.format(
                        self.global_iter, recon_loss.data, combine_sup_loss.data, combine_unsup_loss.data))

                    if self.viz_on:
                        self.gather.insert(images=A_img.data)
                        self.gather.insert(images=B_img.data)
                        self.gather.insert(images=C_img.data)
                        self.gather.insert(images=D_img.data)
                        self.gather.insert(images=E_img.data)
                        self.gather.insert(images=F_img.data)
                        self.gather.insert(images=F.sigmoid(A_recon).data)
                        self.fonts_viz_reconstruction()
                        self.viz_lines()
                        '''
                        combine show
                        '''
                        self.gather.insert(combine_supimages=F.sigmoid(AoC1_2A).data)
                        self.gather.insert(combine_supimages=F.sigmoid(BoC2_2B).data)
                        self.gather.insert(combine_supimages=F.sigmoid(D3Co_2C).data)
                        self.gather.insert(combine_supimages=F.sigmoid(DoC3_2D).data)
                        self.gather.insert(combine_supimages=F.sigmoid(EoC4_2E).data)
                        self.gather.insert(combine_supimages=F.sigmoid(FoC5_2F).data)
                        self.fonts_viz_combine_recon()

                        self.gather.insert(combine_unsupimages=F.sigmoid(A1B2D3E4F5_2C).data)
                        self.gather.insert(combine_unsupimages=F.sigmoid(A2B3D4E5F1_2N).data)
                        self.fonts_viz_combine_unsuprecon()
                        # self.viz_combine(x)
                        self.gather.flush()
                # Save model checkpoints.
                if self.global_iter % self.save_step == 0:
                    Auto_path = os.path.join(self.model_save_dir, self.viz_name,
                                             '{}-Auto.ckpt'.format(self.global_iter))
                    torch.save(self.Autoencoder.state_dict(), Auto_path)
                    print('Saved model checkpoints into {}/{}...'.format(self.model_save_dir, self.viz_name))

                if self.global_iter >= self.max_iter:
                    out = True
                    break

        pbar.write("[Training Finished]")
        pbar.close()
    def test_fonts(self):
        # self.net_mode(train=True)
        # load pretrained model
        self.restore_model('pretrained')
        for index, sup_package in enumerate(self.data_loader):

            A_img = sup_package['A']
            B_img = sup_package['B']
            D_img = sup_package['D']
            E_img = sup_package['E']
            F_img = sup_package['F']

            A_img = Variable(cuda(A_img, self.use_cuda))
            B_img = Variable(cuda(B_img, self.use_cuda))
            D_img = Variable(cuda(D_img, self.use_cuda))
            E_img = Variable(cuda(E_img, self.use_cuda))
            F_img = Variable(cuda(F_img, self.use_cuda))

            ## 1. A B C seperate(first400: id last600 background)
            A_recon, A_z = self.Autoencoder(A_img)
            B_recon, B_z = self.Autoencoder(B_img)
            D_recon, D_z = self.Autoencoder(D_img)
            E_recon, E_z = self.Autoencoder(E_img)
            F_recon, F_z = self.Autoencoder(F_img)
            ''' refer 1: content, 2: size, 3: font-color, 4 back_color, 5 style'''


            A_z_2 = A_z[:, self.z_size_start_dim: self.z_font_color_start_dim]  # 20-40
            B_z_3 = B_z[:, self.z_font_color_start_dim: self.z_back_color_start_dim]  # 40-60
            D_z_4 = D_z[:, self.z_back_color_start_dim: self.z_style_start_dim]  # 60-80
            E_z_5 = E_z[:, self.z_style_start_dim:]  # 80-100
            F_z_1 = F_z[:, 0:self.z_size_start_dim]  # 0-20



            # '''  need unsupervise '''
            A2B3D4E5F1_combine_2N = torch.cat((F_z_1, A_z_2, B_z_3, D_z_4, E_z_5), dim=1)
            mid_A2B3D4E5F1 = self.Autoencoder.fc_decoder(A2B3D4E5F1_combine_2N)
            mid_A2B3D4E5F1 = mid_A2B3D4E5F1.view(A2B3D4E5F1_combine_2N.shape[0], 256, 8, 8)
            A2B3D4E5F1_2N = self.Autoencoder.decoder(mid_A2B3D4E5F1)


            # save synthesized image
            self.test_iter = index
            self.gather.insert(test=F.sigmoid(A2B3D4E5F1_2N).data)
            self.fonts_viz_test()
            self.gather.flush()

    def fonts_save_sample_img(self, tensor, mode):
        unloader = transforms.ToPILImage()
        dir = os.path.join(self.model_save_dir, self.viz_name, 'sample_img')
        if not os.path.exists(dir):
            os.makedirs(dir)
        image = tensor.cpu().clone()  # we clone the tensor to not do changes on it

        if mode == 'recon':
            image_ori_A = image[0].squeeze(0)  # remove the fake batch dimension
            image_ori_B = image[1].squeeze(0)
            image_ori_C = image[2].squeeze(0)
            image_ori_D = image[3].squeeze(0)
            image_ori_E = image[4].squeeze(0)
            image_ori_F = image[5].squeeze(0)
            image_recon = image[6].squeeze(0)

            image_ori_A = unloader(image_ori_A)
            image_ori_B = unloader(image_ori_B)
            image_ori_C = unloader(image_ori_C)
            image_ori_D = unloader(image_ori_D)
            image_ori_E = unloader(image_ori_E)
            image_ori_F = unloader(image_ori_F)
            image_recon = unloader(image_recon)

            image_ori_A.save(os.path.join(dir, '{}-A_img.png'.format(self.global_iter)))
            image_ori_B.save(os.path.join(dir, '{}-B_img.png'.format(self.global_iter)))
            image_ori_C.save(os.path.join(dir, '{}-C_img.png'.format(self.global_iter)))
            image_ori_D.save(os.path.join(dir, '{}-D_img.png'.format(self.global_iter)))
            image_ori_E.save(os.path.join(dir, '{}-E_img.png'.format(self.global_iter)))
            image_ori_F.save(os.path.join(dir, '{}-F_img.png'.format(self.global_iter)))
            image_recon.save(os.path.join(dir, '{}-A_img_recon.png'.format(self.global_iter)))
        elif mode == 'combine_sup':

            image_AoC1_2A = image[0].squeeze(0)  # remove the fake batch dimension
            image_BoC2_2B = image[1].squeeze(0)
            image_D3Co_2C = image[2].squeeze(0)
            image_DoC3_2D = image[3].squeeze(0)
            image_EoC4_2E = image[4].squeeze(0)
            image_FoC5_2F = image[5].squeeze(0)

            image_AoC1_2A = unloader(image_AoC1_2A)
            image_BoC2_2B = unloader(image_BoC2_2B)
            image_D3Co_2C = unloader(image_D3Co_2C)
            image_DoC3_2D = unloader(image_DoC3_2D)
            image_EoC4_2E = unloader(image_EoC4_2E)
            image_FoC5_2F = unloader(image_FoC5_2F)

            image_AoC1_2A.save(os.path.join(dir, '{}-AoC1_2A.png'.format(self.global_iter)))
            image_BoC2_2B.save(os.path.join(dir, '{}-BoC2_2B.png'.format(self.global_iter)))
            image_D3Co_2C.save(os.path.join(dir, '{}-D3Co_2C.png'.format(self.global_iter)))
            image_DoC3_2D.save(os.path.join(dir, '{}-DoC3_2D.png'.format(self.global_iter)))
            image_EoC4_2E.save(os.path.join(dir, '{}-EoC4_2E.png'.format(self.global_iter)))
            image_FoC5_2F.save(os.path.join(dir, '{}-FoC5_2F.png'.format(self.global_iter)))

        elif mode == 'combine_unsup':
            image_A1B2D3E4F5_2C = image[0].squeeze(0)  # remove the fake batch dimension
            image_A2B3D4E5F1_2N = image[1].squeeze(0)

            image_A1B2D3E4F5_2C = unloader(image_A1B2D3E4F5_2C)
            image_A2B3D4E5F1_2N = unloader(image_A2B3D4E5F1_2N)

            image_A1B2D3E4F5_2C.save(os.path.join(dir, '{}-A1B2D3E4F5_2C.png'.format(self.global_iter)))
            image_A2B3D4E5F1_2N.save(os.path.join(dir, '{}-A2B3D4E5F1_2N.png'.format(self.global_iter)))
        elif mode == 'test':
            image_A2B3D4E5F1_2N = image
            image_A2B3D4E5F1_2N = unloader(image_A2B3D4E5F1_2N)
            image_A2B3D4E5F1_2N.save(os.path.join(self.output_dir, 'group{}-image_A2B3D4E5F1_2N.png'.format(self.test_iter)))
    def fonts_viz_reconstruction(self):
        # self.net_mode(train=False)
        x_A = self.gather.data['images'][0][:100]
        x_A = make_grid(x_A, normalize=True)
        x_B = self.gather.data['images'][1][:100]
        x_B = make_grid(x_B, normalize=True)
        x_C = self.gather.data['images'][2][:100]
        x_C = make_grid(x_C, normalize=True)
        x_D = self.gather.data['images'][3][:100]
        x_D = make_grid(x_D, normalize=True)
        x_E = self.gather.data['images'][4][:100]
        x_E = make_grid(x_E, normalize=True)
        x_F = self.gather.data['images'][5][:100]
        x_F = make_grid(x_F, normalize=True)
        x_A_recon = self.gather.data['images'][6][:100]
        x_A_recon = make_grid(x_A_recon, normalize=True)
        images = torch.stack([x_A, x_B, x_C, x_D, x_E, x_F, x_A_recon], dim=0).cpu()
        self.viz.images(images, env=self.viz_name + '_reconstruction',
                        opts=dict(title=str(self.global_iter)), nrow=10)
        self.fonts_save_sample_img(images, 'recon')
        # self.net_mode(train=True)
    def fonts_viz_combine_recon(self):
        # self.net_mode(train=False)
        AoC1_2A = self.gather.data['combine_supimages'][0][:100]
        AoC1_2A = make_grid(AoC1_2A, normalize=True)
        BoC2_2B = self.gather.data['combine_supimages'][1][:100]
        BoC2_2B = make_grid(BoC2_2B, normalize=True)
        D3Co_2C = self.gather.data['combine_supimages'][2][:100]
        D3Co_2C = make_grid(D3Co_2C, normalize=True)
        DoC3_2D = self.gather.data['combine_supimages'][3][:100]
        DoC3_2D = make_grid(DoC3_2D, normalize=True)
        EoC4_2E = self.gather.data['combine_supimages'][4][:100]
        EoC4_2E = make_grid(EoC4_2E, normalize=True)
        FoC5_2F = self.gather.data['combine_supimages'][5][:100]
        FoC5_2F = make_grid(FoC5_2F, normalize=True)
        images = torch.stack([AoC1_2A, BoC2_2B, D3Co_2C, DoC3_2D, EoC4_2E, FoC5_2F], dim=0).cpu()
        self.viz.images(images, env=self.viz_name + 'combine_supimages',
                        opts=dict(title=str(self.global_iter)), nrow=10)
        self.fonts_save_sample_img(images, 'combine_sup')
    def fonts_viz_combine_unsuprecon(self):
        # self.net_mode(train=False)
        A1B2D3E4F5_2C = self.gather.data['combine_unsupimages'][0][:100]
        A1B2D3E4F5_2C = make_grid(A1B2D3E4F5_2C, normalize=True)
        A2B3D4E5F1_2N = self.gather.data['combine_unsupimages'][1][:100]
        A2B3D4E5F1_2N = make_grid(A2B3D4E5F1_2N, normalize=True)
        images = torch.stack([A1B2D3E4F5_2C, A2B3D4E5F1_2N], dim=0).cpu()
        self.viz.images(images, env=self.viz_name + 'combine_unsupimages',
                        opts=dict(title=str(self.global_iter)), nrow=10)
        self.fonts_save_sample_img(images, 'combine_unsup')
    def fonts_viz_test(self):
        # self.net_mode(train=False)
        A2B3D4E5F1_2N = self.gather.data['test'][0][:100]
        A2B3D4E5F1_2N = make_grid(A2B3D4E5F1_2N, normalize=True)
        images = A2B3D4E5F1_2N
        self.fonts_save_sample_img(images, 'test')

    # For RaFD dataset
    def train_rafd(self):
        # self.net_mode(train=True)
        out = False
        # Start training from scratch or resume training.
        self.global_iter = 0
        if self.resume_iters:
            self.global_iter = self.resume_iters
            self.restore_model(self.resume_iters)

        pbar = tqdm(total=self.max_iter)
        pbar.update(self.global_iter)
        while not out:
            for sup_package in self.data_loader:
                #print(self.global_iter)
                # id, expression, pose
                A_img = sup_package['A']
                B_img = sup_package['B']
                C_img = sup_package['C']
                D_img = sup_package['D']
                E_img = sup_package['E']
                self.global_iter += 1
                pbar.update(1)

                A_img = Variable(cuda(A_img, self.use_cuda))
                B_img = Variable(cuda(B_img, self.use_cuda))
                C_img = Variable(cuda(C_img, self.use_cuda))
                D_img = Variable(cuda(D_img, self.use_cuda))
                E_img = Variable(cuda(E_img, self.use_cuda))

                ## 1. A B C seperate(first400: id last600 background)
                A_recon, A_z = self.Autoencoder(A_img)
                B_recon, B_z = self.Autoencoder(B_img)
                C_recon, C_z = self.Autoencoder(C_img)
                D_recon, D_z = self.Autoencoder(D_img)
                E_recon, E_z = self.Autoencoder(E_img)

                A_z_id = A_z[:, 0:self.z_id_dim] # 0-700
                A_z_expression = A_z[:, self.z_id_dim:self.z_id_dim + self.z_expression_dim] # 700-800
                A_z_pose = A_z[:, self.z_id_dim + self.z_expression_dim:] #800-1000
                B_z_id = B_z[:, 0:self.z_id_dim]
                B_z_expression = B_z[:, self.z_id_dim:self.z_id_dim + self.z_expression_dim]
                B_z_pose = B_z[:, self.z_id_dim + self.z_expression_dim:]
                C_z_id = C_z[:, 0:self.z_id_dim]
                C_z_expression = C_z[:, self.z_id_dim:self.z_id_dim + self.z_expression_dim]
                C_z_pose = C_z[:, self.z_id_dim + self.z_expression_dim:]
                D_z_id = D_z[:, 0:self.z_id_dim]
                D_z_expression = D_z[:, self.z_id_dim:self.z_id_dim + self.z_expression_dim]
                D_z_pose = D_z[:, self.z_id_dim + self.z_expression_dim:]
                E_z_id = E_z[:, 0:self.z_id_dim]
                E_z_expression = E_z[:, self.z_id_dim:self.z_id_dim + self.z_expression_dim]
                E_z_pose = E_z[:, self.z_id_dim + self.z_expression_dim:]

                ## unsup cycle,A, E
                # id
                AiEuEp_z = torch.cat((A_z_id, E_z_expression), dim=1)
                AiEuEp_z = torch.cat((AiEuEp_z, E_z_pose), dim=1)
                EiAuAp_z = torch.cat((E_z_id, A_z_expression), dim=1)
                EiAuAp_z = torch.cat((EiAuAp_z, A_z_pose), dim=1)

                mid_AiEuEp = self.Autoencoder.fc_decoder(AiEuEp_z)
                mid_EiAuAp = self.Autoencoder.fc_decoder(EiAuAp_z)
                mid_AiEuEp = mid_AiEuEp.view(AiEuEp_z.shape[0], 256, 8, 8)
                mid_EiAuAp = mid_EiAuAp.view(EiAuAp_z.shape[0], 256, 8, 8)
                AiEuEp = self.Autoencoder.decoder(mid_AiEuEp)
                EiAuAp = self.Autoencoder.decoder(mid_EiAuAp)

                AiEuEp_recon, AiEuEp_z1 = self.Autoencoder(AiEuEp)
                EiAuAp_recon, EiAuAp_z1 = self.Autoencoder(EiAuAp)

                AiEuEp_z_id = AiEuEp_z1[:, 0:self.z_id_dim]  # 0-700
                AiEuEp_z_expression = AiEuEp_z1[:, self.z_id_dim:self.z_id_dim + self.z_expression_dim]  # 700-800
                AiEuEp_z_pose = AiEuEp_z1[:, self.z_id_dim + self.z_expression_dim:]  # 800-1000
                EiAuAp_z_id = EiAuAp_z1[:, 0:self.z_id_dim]  # 0-700
                EiAuAp_z_expression = EiAuAp_z1[:, self.z_id_dim:self.z_id_dim + self.z_expression_dim]  # 700-800
                EiAuAp_z_pose = EiAuAp_z1[:, self.z_id_dim + self.z_expression_dim:]  # 800-1000

                AiEuEp_z2 = torch.cat((AiEuEp_z_id, EiAuAp_z_expression), dim=1)
                AiEuEp_z2 = torch.cat((AiEuEp_z2, EiAuAp_z_pose), dim=1)
                EiAuAp_z2 = torch.cat((EiAuAp_z_id, AiEuEp_z_expression), dim=1)
                EiAuAp_z2 = torch.cat((EiAuAp_z2, AiEuEp_z_pose), dim=1)

                mid_A2 = self.Autoencoder.fc_decoder(AiEuEp_z2)
                mid_E2 = self.Autoencoder.fc_decoder(EiAuAp_z2)
                mid_A2 = mid_A2.view(AiEuEp_z2.shape[0], 256, 8, 8)
                mid_E2 = mid_E2.view(EiAuAp_z2.shape[0], 256, 8, 8)
                A2 = self.Autoencoder.decoder(mid_A2)
                E2 = self.Autoencoder.decoder(mid_E2)

                # uk
                AiEuAp_z = torch.cat((A_z_id, E_z_expression), dim=1)
                AiEuAp_z = torch.cat((AiEuAp_z, A_z_pose), dim=1)
                EiAuEp_z = torch.cat((E_z_id, A_z_expression), dim=1)
                EiAuEp_z = torch.cat((EiAuEp_z, E_z_pose), dim=1)

                mid_AiEuAp = self.Autoencoder.fc_decoder(AiEuAp_z)
                mid_EiAuEp = self.Autoencoder.fc_decoder(EiAuEp_z)
                mid_AiEuAp = mid_AiEuAp.view(AiEuAp_z.shape[0], 256, 8, 8)
                mid_EiAuEp = mid_EiAuEp.view(EiAuEp_z.shape[0], 256, 8, 8)
                AiEuAp = self.Autoencoder.decoder(mid_AiEuAp)
                EiAuEp = self.Autoencoder.decoder(mid_EiAuEp)

                AiEuAp_recon, AiEuAp_z1 = self.Autoencoder(AiEuAp)
                EiAuEp_recon, EiAuEp_z1 = self.Autoencoder(EiAuEp)

                AiEuAp_z_id = AiEuAp_z1[:, 0:self.z_id_dim]  # 0-700
                AiEuAp_z_expression = AiEuAp_z1[:, self.z_id_dim:self.z_id_dim + self.z_expression_dim]  # 700-800
                AiEuAp_z_pose = AiEuAp_z1[:, self.z_id_dim + self.z_expression_dim:]  # 800-1000
                EiAuEp_z_id = EiAuEp_z1[:, 0:self.z_id_dim]  # 0-700
                EiAuEp_z_expression = EiAuEp_z1[:, self.z_id_dim:self.z_id_dim + self.z_expression_dim]  # 700-800
                EiAuEp_z_pose = EiAuEp_z1[:, self.z_id_dim + self.z_expression_dim:]  # 800-1000

                AiEuAp_z2 = torch.cat((AiEuAp_z_id, EiAuEp_z_expression), dim=1)
                AiEuAp_z2 = torch.cat((AiEuAp_z2, AiEuAp_z_pose), dim=1)
                EiAuEp_z2 = torch.cat((EiAuEp_z_id, AiEuAp_z_expression), dim=1)
                EiAuEp_z2 = torch.cat((EiAuEp_z2, EiAuEp_z_pose), dim=1)

                mid_A3 = self.Autoencoder.fc_decoder(AiEuAp_z2)
                mid_E3 = self.Autoencoder.fc_decoder(EiAuEp_z2)
                mid_A3 = mid_A3.view(AiEuAp_z2.shape[0], 256, 8, 8)
                mid_E3 = mid_E3.view(EiAuEp_z2.shape[0], 256, 8, 8)
                A3 = self.Autoencoder.decoder(mid_A3)
                E3 = self.Autoencoder.decoder(mid_E3)

                # pz
                AiAuEp_z = torch.cat((A_z_id, A_z_expression), dim=1)
                AiAuEp_z= torch.cat((AiAuEp_z, E_z_pose), dim=1)
                EiEuAp_z = torch.cat((E_z_id, E_z_expression), dim=1)
                EiEuAp_z = torch.cat((EiEuAp_z, A_z_pose), dim=1)

                mid_AiAuEp = self.Autoencoder.fc_decoder(AiAuEp_z)
                mid_EiEuAp = self.Autoencoder.fc_decoder(EiEuAp_z)
                mid_AiAuEp = mid_AiAuEp.view(AiAuEp_z.shape[0], 256, 8, 8)
                mid_EiEuAp = mid_EiEuAp.view(EiEuAp_z.shape[0], 256, 8, 8)
                AiAuEp = self.Autoencoder.decoder(mid_AiAuEp)
                EiEuAp = self.Autoencoder.decoder(mid_EiEuAp)

                AiAuEp_recon, AiAuEp_z1 = self.Autoencoder(AiAuEp)
                EiEuAp_recon, EiEuAp_z1 = self.Autoencoder(EiEuAp)

                AiAuEp_z_id = AiAuEp_z1[:, 0:self.z_id_dim]  # 0-700
                AiAuEp_z_expression = AiAuEp_z1[:, self.z_id_dim:self.z_id_dim + self.z_expression_dim]  # 700-800
                AiAuEp_z_pose = AiAuEp_z1[:, self.z_id_dim + self.z_expression_dim:]  # 800-1000
                EiEuAp_z_id = EiEuAp_z1[:, 0:self.z_id_dim]  # 0-700
                EiEuAp_z_expression = EiEuAp_z1[:, self.z_id_dim:self.z_id_dim + self.z_expression_dim]  # 700-800
                EiEuAp_z_pose = EiEuAp_z1[:, self.z_id_dim + self.z_expression_dim:]  # 800-1000

                AiAuEp_z2 = torch.cat((AiAuEp_z_id, AiAuEp_z_expression), dim=1)
                AiAuEp_z2 = torch.cat((AiAuEp_z2, EiEuAp_z_pose), dim=1)
                EiEuAp_z2 = torch.cat((EiEuAp_z_id, EiEuAp_z_expression), dim=1)
                EiEuAp_z2 = torch.cat((EiEuAp_z2, AiAuEp_z_pose), dim=1)

                mid_A4 = self.Autoencoder.fc_decoder(AiAuEp_z2)
                mid_E4 = self.Autoencoder.fc_decoder(EiEuAp_z2)
                mid_A4 = mid_A4.view(AiAuEp_z2.shape[0], 256, 8, 8)
                mid_E4 = mid_E4.view(EiEuAp_z2.shape[0], 256, 8, 8)
                A4 = self.Autoencoder.decoder(mid_A4)
                E4 = self.Autoencoder.decoder(mid_E4)

                ## 2. combine with strong supervise
                # C A same pose diff id, back
                ApCo_combine_2C = torch.cat((C_z_id, C_z_expression), dim=1)
                ApCo_combine_2C = torch.cat((ApCo_combine_2C, A_z_pose), dim=1)
                mid_ApCo = self.Autoencoder.fc_decoder(ApCo_combine_2C)
                mid_ApCo = mid_ApCo.view(ApCo_combine_2C.shape[0], 256, 8, 8)
                ApCo_2C = self.Autoencoder.decoder(mid_ApCo)

                AoCp_combine_2A = torch.cat((A_z_id, A_z_expression), dim=1)
                AoCp_combine_2A = torch.cat((AoCp_combine_2A, C_z_pose), dim=1)
                mid_AoCp = self.Autoencoder.fc_decoder(AoCp_combine_2A)
                mid_AoCp = mid_AoCp.view(AoCp_combine_2A.shape[0], 256, 8, 8)
                AoCp_2A = self.Autoencoder.decoder(mid_AoCp)

                # C B same id diff pose, back
                BaCo_combine_2C = torch.cat((B_z_id, C_z_expression), dim=1)
                BaCo_combine_2C = torch.cat((BaCo_combine_2C, C_z_pose), dim=1)
                mid_BaCo = self.Autoencoder.fc_decoder(BaCo_combine_2C)
                mid_BaCo = mid_BaCo.view(BaCo_combine_2C.shape[0], 256, 8, 8)
                BaCo_2C = self.Autoencoder.decoder(mid_BaCo)

                BoCa_combine_2B = torch.cat((C_z_id, B_z_expression), dim=1)
                BoCa_combine_2B = torch.cat((BoCa_combine_2B, B_z_pose), dim=1)
                mid_BoCa = self.Autoencoder.fc_decoder(BoCa_combine_2B)
                mid_BoCa = mid_BoCa.view(BoCa_combine_2B.shape[0], 256, 8, 8)
                BoCa_2B = self.Autoencoder.decoder(mid_BoCa)

                # C D same background diff id, pose
                DbCo_combine_2C = torch.cat((C_z_id, D_z_expression), dim=1)
                DbCo_combine_2C = torch.cat((DbCo_combine_2C, C_z_pose), dim=1)
                mid_DbCo = self.Autoencoder.fc_decoder(DbCo_combine_2C)
                mid_DbCo = mid_DbCo.view(DbCo_combine_2C.shape[0], 256, 8, 8)
                DbCo_2C = self.Autoencoder.decoder(mid_DbCo)

                DoCb_combine_2D = torch.cat((D_z_id, C_z_expression), dim=1)
                DoCb_combine_2D = torch.cat((DoCb_combine_2D, D_z_pose), dim=1)
                mid_DoCb = self.Autoencoder.fc_decoder(DoCb_combine_2D)
                mid_DoCb = mid_DoCb.view(DoCb_combine_2D.shape[0], 256, 8, 8)
                DoCb_2D = self.Autoencoder.decoder(mid_DoCb)

                # combine_2C
                ApBaDb_combine_2C = torch.cat((B_z_id, D_z_expression), dim=1)
                ApBaDb_combine_2C = torch.cat((ApBaDb_combine_2C, A_z_pose), dim=1)
                mid_ApBaDb = self.Autoencoder.fc_decoder(ApBaDb_combine_2C)
                mid_ApBaDb = mid_ApBaDb.view(ApBaDb_combine_2C.shape[0], 256, 8, 8)
                ApBaDb_2C = self.Autoencoder.decoder(mid_ApBaDb)



                # '''  need unsupervise '''
                AaBpDb_combine_2N = torch.cat((A_z_id, D_z_expression), dim=1)
                AaBpDb_combine_2N = torch.cat((AaBpDb_combine_2N, B_z_pose), dim=1)
                mid_AaBpDb = self.Autoencoder.fc_decoder(AaBpDb_combine_2N)
                mid_AaBpDb = mid_AaBpDb.view(AaBpDb_combine_2N.shape[0], 256, 8, 8)
                AaBpDb_2N = self.Autoencoder.decoder(mid_AaBpDb)

                # '''  need unsupervise '''
                # AaBp_combine_2N = torch.cat((A_z_id, C_z_expression), dim=1)
                # AaBp_combine_2N = torch.cat((AaBp_combine_2N, B_z_pose), dim=1)
                # mid_AaBp = self.Autoencoder.fc_decoder(AaBp_combine_2N)
                # mid_AaBp = mid_AaBp.view(AaBp_combine_2N.shape[0], 256, 8, 8)
                # AaBp_2N = self.Autoencoder.decoder(mid_AaBp)


                '''
                optimize for autoencoder
                '''

                # 1. recon_loss
                A_recon_loss = torch.mean(torch.abs(A_img - A_recon))
                B_recon_loss = torch.mean(torch.abs(B_img - B_recon))
                C_recon_loss = torch.mean(torch.abs(C_img - C_recon))
                D_recon_loss = torch.mean(torch.abs(D_img - D_recon))
                recon_loss = A_recon_loss + B_recon_loss + C_recon_loss + D_recon_loss

                # 2. sup_combine_loss
                ApCo_2C_loss = torch.mean(torch.abs(C_img - ApCo_2C))
                AoCp_2A_loss = torch.mean(torch.abs(A_img - AoCp_2A))
                BaCo_2C_loss = torch.mean(torch.abs(C_img - BaCo_2C))
                BoCa_2B_loss = torch.mean(torch.abs(B_img - BoCa_2B))
                DbCo_2C_loss = torch.mean(torch.abs(C_img - DbCo_2C))
                DoCb_2D_loss = torch.mean(torch.abs(D_img - DoCb_2D))
                ApBaDb_2C_loss = torch.mean(torch.abs(C_img - ApBaDb_2C))
                combine_sup_loss = ApCo_2C_loss + AoCp_2A_loss + BaCo_2C_loss + BoCa_2B_loss + DbCo_2C_loss + DoCb_2D_loss + ApBaDb_2C_loss

                # 3. unsup_combine_loss
                _, AaBpDb_z = self.Autoencoder(AaBpDb_2N)
                combine_unsup_loss = torch.mean(torch.abs(A_z_id - AaBpDb_z[:, 0:self.z_id_dim])) + torch.mean(torch.abs(D_z_expression - AaBpDb_z[:, self.z_id_dim:self.z_id_dim + self.z_expression_dim])) + torch.mean(torch.abs(B_z_pose - AaBpDb_z[:, self.z_id_dim + self.z_expression_dim:]))

                # 4. unsup cycle loss
                A2_loss = torch.mean(torch.abs(A_img - A2))
                A3_loss = torch.mean(torch.abs(A_img - A3))
                A4_loss = torch.mean(torch.abs(A_img - A4))
                E2_loss = torch.mean(torch.abs(E_img - E2))
                E3_loss = torch.mean(torch.abs(E_img - E3))
                E4_loss = torch.mean(torch.abs(E_img - E4))

                cycle_loss = A2_loss + A3_loss + A4_loss + E2_loss + E3_loss+ E4_loss
                #
                # whole loss
                vae_unsup_loss = recon_loss + self.lambda_combine * combine_sup_loss + self.lambda_unsup * combine_unsup_loss  #+ self.lambda_cycle * cycle_loss
                self.auto_optim.zero_grad()
                vae_unsup_loss.backward()
                self.auto_optim.step()

                #　save the log
                f = open(self.log_dir + '/log.txt', 'a')
                f.writelines(['\n', '[{}] recon_loss:{:.3f}  combine_sup_loss:{:.3f}  combine_unsup_loss:{:.3f} cycled_loss:{:.3f}'.format(
                        self.global_iter, recon_loss.data, combine_sup_loss.data, combine_unsup_loss.data, cycle_loss.data)])
                f.close()
                print('\n', '[{}] recon_loss:{:.3f}  combine_sup_loss:{:.3f}  combine_unsup_loss:{:.3f} cycled_loss:{:.3f}'.format(
                        self.global_iter, recon_loss.data, combine_sup_loss.data, combine_unsup_loss.data, cycle_loss.data))


                if self.viz_on and self.global_iter%self.gather_step == 0:
                    self.gather.insert(iter=self.global_iter,recon_loss=recon_loss.data,
                                    combine_sup_loss=combine_sup_loss.data, combine_unsup_loss=combine_unsup_loss.data)

                if self.global_iter%self.display_step == 0:
                    pbar.write('[{}] recon_loss:{:.3f}  combine_sup_loss:{:.3f}  combine_unsup_loss:{:.3f} cycle_loss:{:.3f}'.format(
                        self.global_iter, recon_loss.data, combine_sup_loss.data, combine_unsup_loss.data, cycle_loss.data))

                    if self.viz_on:
                        self.gather.insert(images=A_img.data)
                        self.gather.insert(images=B_img.data)
                        self.gather.insert(images=C_img.data)
                        self.gather.insert(images=D_img.data)
                        self.gather.insert(images=F.sigmoid(A_recon).data)
                        self.rafd_viz_reconstruction()
                        self.viz_lines()
                        '''
                        combine show
                        '''
                        self.gather.insert(combine_supimages=F.sigmoid(AoCp_2A).data)
                        self.gather.insert(combine_supimages=F.sigmoid(BoCa_2B).data)
                        self.gather.insert(combine_supimages=F.sigmoid(DbCo_2C).data)
                        self.gather.insert(combine_supimages=F.sigmoid(DoCb_2D).data)
                        self.rafd_viz_combine_recon()

                        self.gather.insert(combine_unsupimages=F.sigmoid(ApBaDb_2C).data)
                        self.gather.insert(combine_unsupimages=F.sigmoid(AaBpDb_2N).data)
                        self.gather.insert(combine_unsupimages=E_img.data)
                        self.gather.insert(combine_unsupimages=AiEuEp.data)
                        self.gather.insert(combine_unsupimages=EiAuAp.data)
                        self.gather.insert(combine_unsupimages=AiEuAp.data)
                        self.gather.insert(combine_unsupimages=EiAuEp.data)
                        self.gather.insert(combine_unsupimages=AiAuEp.data)
                        self.gather.insert(combine_unsupimages=EiEuAp.data)
                        self.gather.insert(combine_unsupimages=A2.data)
                        self.gather.insert(combine_unsupimages=A3.data)
                        self.gather.insert(combine_unsupimages=A4.data)
                        self.gather.insert(combine_unsupimages=E2.data)
                        self.gather.insert(combine_unsupimages=E3.data)
                        self.gather.insert(combine_unsupimages=E4.data)

                        self.rafd_viz_combine_unsuprecon()
                        # self.viz_combine(x)
                        self.gather.flush()
                # Save model checkpoints.
                if self.global_iter%self.save_step == 0:
                    Auto_path = os.path.join(self.model_save_dir, self.viz_name, '{}-Auto.ckpt'.format(self.global_iter))
                    torch.save(self.Autoencoder.state_dict(), Auto_path)
                    print('Saved model checkpoints into {}/{}...'.format(self.model_save_dir, self.viz_name))


                if self.global_iter >= self.max_iter:
                    out = True
                    break

        pbar.write("[Training Finished]")
        pbar.close()
    def test_rafd(self):
        # self.net_mode(train=True)
        # load pretrained model
        self.restore_model('pretrained')
        for index, sup_package in enumerate(self.data_loader):
            # id, expression, pose
            A_img = sup_package['A']
            B_img = sup_package['B']
            D_img = sup_package['D']

            A_img = Variable(cuda(A_img, self.use_cuda))
            B_img = Variable(cuda(B_img, self.use_cuda))
            D_img = Variable(cuda(D_img, self.use_cuda))

            ## 1. A B C get latent
            A_recon, A_z = self.Autoencoder(A_img)
            B_recon, B_z = self.Autoencoder(B_img)
            D_recon, D_z = self.Autoencoder(D_img)

            A_z_id = A_z[:, 0:self.z_id_dim] # 0-60
            B_z_pose = B_z[:, self.z_id_dim + self.z_expression_dim:]
            D_z_expression = D_z[:, self.z_id_dim:self.z_id_dim + self.z_expression_dim]

            ## 2. combine with strong supervise

            AaBpDb_combine_2N = torch.cat((A_z_id, D_z_expression), dim=1)
            AaBpDb_combine_2N = torch.cat((AaBpDb_combine_2N, B_z_pose), dim=1)
            mid_AaBpDb = self.Autoencoder.fc_decoder(AaBpDb_combine_2N)
            mid_AaBpDb = mid_AaBpDb.view(AaBpDb_combine_2N.shape[0], 256, 8, 8)
            AaBpDb_2N = self.Autoencoder.decoder(mid_AaBpDb)

            # save synthesized image
            self.test_iter = index
            self.gather.insert(test=F.sigmoid(AaBpDb_2N).data)
            self.ilab20m_viz_test()
            self.gather.flush()





    def rafd_save_sample_img(self, tensor, mode):
        unloader = transforms.ToPILImage()
        dir = os.path.join(self.model_save_dir, self.viz_name, 'sample_img')
        if not os.path.exists(dir):
            os.makedirs(dir)
        image = tensor.cpu().clone()  # we clone the tensor to not do changes on it

        if mode == 'recon':
            image_ori_A = image[0].squeeze(0)  # remove the fake batch dimension
            image_ori_B = image[1].squeeze(0)
            image_ori_C = image[2].squeeze(0)
            image_ori_D = image[3].squeeze(0)
            image_recon = image[4].squeeze(0)

            image_ori_A = unloader(image_ori_A)
            image_ori_B = unloader(image_ori_B)
            image_ori_C = unloader(image_ori_C)
            image_ori_D = unloader(image_ori_D)
            image_recon = unloader(image_recon)

            image_ori_A.save(os.path.join(dir, '{}-A_img.png'.format(self.global_iter)))
            image_ori_B.save(os.path.join(dir, '{}-B_img.png'.format(self.global_iter)))
            image_ori_C.save(os.path.join(dir, '{}-C_img.png'.format(self.global_iter)))
            image_ori_D.save(os.path.join(dir, '{}-D_img.png'.format(self.global_iter)))
            image_recon.save(os.path.join(dir, '{}-A_img_recon.png'.format(self.global_iter)))
        elif mode == 'combine_sup':
            image_AoCp_2A = image[0].squeeze(0)  # remove the fake batch dimension
            image_BoCa_2B = image[1].squeeze(0)
            image_DbCo_2C = image[2].squeeze(0)
            image_DoCb_2D = image[3].squeeze(0)

            image_AoCp_2A = unloader(image_AoCp_2A)
            image_BoCa_2B = unloader(image_BoCa_2B)
            image_DbCo_2C = unloader(image_DbCo_2C)
            image_DoCb_2D = unloader(image_DoCb_2D)

            image_AoCp_2A.save(os.path.join(dir, '{}-AoCp_2A.png'.format(self.global_iter)))
            image_BoCa_2B.save(os.path.join(dir, '{}-BoCa_2B.png'.format(self.global_iter)))
            image_DbCo_2C.save(os.path.join(dir, '{}-DbCo_2C.png'.format(self.global_iter)))
            image_DoCb_2D.save(os.path.join(dir, '{}-DoCb_2D.png'.format(self.global_iter)))
        elif mode == 'combine_unsup':
            image_ApBaDb_2C = image[0].squeeze(0)  # remove the fake batch dimension
            image_AaBpDb_2N = image[1].squeeze(0)
            image_E = image[2].squeeze(0)
            image_AiEuEp = image[3].squeeze(0)
            image_EiAuAp = image[4].squeeze(0)
            image_AiEuAp = image[5].squeeze(0)
            image_EiAuEp = image[6].squeeze(0)
            image_AiAuEp = image[7].squeeze(0)
            image_EiEuAp = image[8].squeeze(0)
            image_A2 = image[9].squeeze(0)
            image_A3 = image[10].squeeze(0)
            image_A4 = image[11].squeeze(0)
            image_E2 = image[12].squeeze(0)
            image_E3 = image[13].squeeze(0)
            image_E4 = image[14].squeeze(0)

            image_ApBaDb_2C = unloader(image_ApBaDb_2C)
            image_AaBpDb_2N = unloader(image_AaBpDb_2N)
            image_E = unloader(image_E)
            image_AiEuEp = unloader(image_AiEuEp)
            image_EiAuAp = unloader(image_EiAuAp)
            image_AiEuAp = unloader(image_AiEuAp)
            image_EiAuEp = unloader(image_EiAuEp)
            image_AiAuEp = unloader(image_AiAuEp)
            image_EiEuAp = unloader(image_EiEuAp)
            image_A2 = unloader(image_A2)
            image_A3 = unloader(image_A3)
            image_A4 = unloader(image_A4)
            image_E2 = unloader(image_E2)
            image_E3 = unloader(image_E3)
            image_E4 = unloader(image_E4)

            image_ApBaDb_2C.save(os.path.join(dir, '{}-ApBaDb_2C.png'.format(self.global_iter)))
            image_AaBpDb_2N.save(os.path.join(dir, '{}-AaBpDb_2N.png'.format(self.global_iter)))
            image_E.save(os.path.join(dir, '{}-E.png'.format(self.global_iter)))
            image_AiEuEp.save(os.path.join(dir, '{}-AiEuEp.png'.format(self.global_iter)))
            image_EiAuAp.save(os.path.join(dir, '{}-EiAuAp.png'.format(self.global_iter)))
            image_AiEuAp.save(os.path.join(dir, '{}-AiEuAp.png'.format(self.global_iter)))
            image_EiAuEp.save(os.path.join(dir, '{}-EiAuEp.png'.format(self.global_iter)))
            image_AiAuEp.save(os.path.join(dir, '{}-AiAuEp.png'.format(self.global_iter)))
            image_EiEuAp.save(os.path.join(dir, '{}-EiEuAp.png'.format(self.global_iter)))
            image_A2.save(os.path.join(dir, '{}-A2.png'.format(self.global_iter)))
            image_A3.save(os.path.join(dir, '{}-A3.png'.format(self.global_iter)))
            image_A4.save(os.path.join(dir, '{}-A4.png'.format(self.global_iter)))
            image_E2.save(os.path.join(dir, '{}-E2.png'.format(self.global_iter)))
            image_E3.save(os.path.join(dir, '{}-E3.png'.format(self.global_iter)))
            image_E4.save(os.path.join(dir, '{}-E4.png'.format(self.global_iter)))
    def rafd_viz_reconstruction(self):
        # self.net_mode(train=False)
        x_A = self.gather.data['images'][0][:100]
        x_A = make_grid(x_A, normalize=True)
        x_B = self.gather.data['images'][1][:100]
        x_B = make_grid(x_B, normalize=True)
        x_C = self.gather.data['images'][2][:100]
        x_C = make_grid(x_C, normalize=True)
        x_D = self.gather.data['images'][3][:100]
        x_D = make_grid(x_D, normalize=True)
        x_A_recon = self.gather.data['images'][4][:100]
        x_A_recon = make_grid(x_A_recon, normalize=True)
        images = torch.stack([x_A, x_B, x_C, x_D, x_A_recon], dim=0).cpu()
        self.viz.images(images, env=self.viz_name+'_reconstruction',
                        opts=dict(title=str(self.global_iter)), nrow=10)
        self.rafd_save_sample_img(images, 'recon')
        # self.net_mode(train=True)
    def rafd_viz_combine_recon(self):
        # self.net_mode(train=False)
        AoCp_2A = self.gather.data['combine_supimages'][0][:100]
        AoCp_2A = make_grid(AoCp_2A, normalize=True)
        BoCa_2B = self.gather.data['combine_supimages'][1][:100]
        BoCa_2B = make_grid(BoCa_2B, normalize=True)
        DbCo_2C = self.gather.data['combine_supimages'][2][:100]
        DbCo_2C = make_grid(DbCo_2C, normalize=True)
        DoCb_2D = self.gather.data['combine_supimages'][3][:100]
        DoCb_2D = make_grid(DoCb_2D, normalize=True)
        images = torch.stack([AoCp_2A, BoCa_2B, DbCo_2C, DoCb_2D], dim=0).cpu()
        self.viz.images(images, env=self.viz_name+'combine_supimages',
                        opts=dict(title=str(self.global_iter)), nrow=10)
        self.rafd_save_sample_img(images, 'combine_sup')
    def rafd_viz_combine_unsuprecon(self):
        # self.net_mode(train=False)
        ApBaDb_2C = self.gather.data['combine_unsupimages'][0][:100]
        ApBaDb_2C = make_grid(ApBaDb_2C, normalize=True)
        AaBpDb_2N = self.gather.data['combine_unsupimages'][1][:100]
        AaBpDb_2N = make_grid(AaBpDb_2N, normalize=True)
        E = self.gather.data['combine_unsupimages'][2][:100]
        E = make_grid(E, normalize=True)
        AiEuEp = self.gather.data['combine_unsupimages'][3][:100]
        AiEuEp = make_grid(AiEuEp, normalize=True)
        EiAuAp = self.gather.data['combine_unsupimages'][4][:100]
        EiAuAp = make_grid(EiAuAp, normalize=True)
        AiEuAp = self.gather.data['combine_unsupimages'][5][:100]
        AiEuAp = make_grid(AiEuAp, normalize=True)
        EiAuEp = self.gather.data['combine_unsupimages'][6][:100]
        EiAuEp = make_grid(EiAuEp, normalize=True)
        AiAuEp = self.gather.data['combine_unsupimages'][7][:100]
        AiAuEp = make_grid(AiAuEp, normalize=True)
        EiEuAp = self.gather.data['combine_unsupimages'][8][:100]
        EiEuAp = make_grid(EiEuAp, normalize=True)
        A2 = self.gather.data['combine_unsupimages'][9][:100]
        A2 = make_grid(A2, normalize=True)
        A3 = self.gather.data['combine_unsupimages'][10][:100]
        A3 = make_grid(A3, normalize=True)
        A4 = self.gather.data['combine_unsupimages'][11][:100]
        A4 = make_grid(A4, normalize=True)
        E2 = self.gather.data['combine_unsupimages'][12][:100]
        E2 = make_grid(E2, normalize=True)
        E3 = self.gather.data['combine_unsupimages'][13][:100]
        E3 = make_grid(E3, normalize=True)
        E4 = self.gather.data['combine_unsupimages'][14][:100]
        E4 = make_grid(E4, normalize=True)

        images = torch.stack([ApBaDb_2C, AaBpDb_2N,E,AiEuEp,EiAuAp,AiEuAp,EiAuEp,AiAuEp,EiEuAp,A2,A3,A4,E2,E3,E4], dim=0).cpu()
        self.viz.images(images, env=self.viz_name+'combine_unsupimages',
                        opts=dict(title=str(self.global_iter)), nrow=10)
        self.rafd_save_sample_img(images, 'combine_unsup')
    def rafd_viz_combine(self, x):
        # self.net_mode(train=False)

        decoder = self.Autoencoder.decoder
        encoder = self.Autoencoder.encoder
        z = encoder(x)
        z_id = z[:, 0:250, :, :]
        z_pose = z[:, 250:, :, :]
        z_rearrange_combine = torch.cat((z_id[:-1], z_pose[1:]), dim=1)
        x_rearrange_combine = decoder(z_rearrange_combine)
        x_rearrange_combine = F.sigmoid(x_rearrange_combine).data

        x_show = make_grid(x[:-1].data, normalize=True)
        x_rearrange_combine_show = make_grid(x_rearrange_combine, normalize=True)
        images = torch.stack([x_show, x_rearrange_combine_show], dim=0).cpu()
        self.viz.images(images, env=self.viz_name+'_combine',
                        opts=dict(title=str(self.global_iter)), nrow=10)

    # For dsprite dataset
    def train_dsprites(self):
        '''
        dsprites has 5 attributes, so the training process is similar as Fonts.
        shape: square, ellipse, heart / scale: 6 values linearly spaced in [0.5, 1]/ Orientation: 40 values in [0, 2 pi] /
        Position X: 32 values in [0, 1] / Position Y: 32 values in [0, 1]
        shape, scale, Orientation, Position X, Position Y
        '''

        out = False
        # Start training from scratch or resume training.
        self.global_iter = 0
        if self.resume_iters:
            self.global_iter = self.resume_iters
            self.restore_model(self.resume_iters)

        pbar = tqdm(total=self.max_iter)
        pbar.update(self.global_iter)
        while not out:
            for sup_package in self.data_loader:
                A_img = sup_package['A']
                B_img = sup_package['B']
                C_img = sup_package['C']
                D_img = sup_package['D']
                E_img = sup_package['E']
                F_img = sup_package['F']
                self.global_iter += 1
                pbar.update(1)

                A_img = Variable(cuda(A_img, self.use_cuda))
                B_img = Variable(cuda(B_img, self.use_cuda))
                C_img = Variable(cuda(C_img, self.use_cuda))
                D_img = Variable(cuda(D_img, self.use_cuda))
                E_img = Variable(cuda(E_img, self.use_cuda))
                F_img = Variable(cuda(F_img, self.use_cuda))

                ## 1. A B C seperate(first400: id last600 background)
                A_recon, A_z = self.Autoencoder(A_img)
                B_recon, B_z = self.Autoencoder(B_img)
                C_recon, C_z = self.Autoencoder(C_img)
                D_recon, D_z = self.Autoencoder(D_img)
                E_recon, E_z = self.Autoencoder(E_img)
                F_recon, F_z = self.Autoencoder(F_img)
                ''' refer 1: shape, 2: scale, 3: Orientation, 4 Position X, 5 Position Y
                    we use the fonts code because both of them have 5 attributes, do not confulsed by the variable name
                '''

                A_z_1 = A_z[:, 0:self.z_size_start_dim] # 0-20
                A_z_2 = A_z[:, self.z_size_start_dim : self.z_font_color_start_dim] # 20-40
                A_z_3 = A_z[:, self.z_font_color_start_dim : self.z_back_color_start_dim] #40-60
                A_z_4 = A_z[:, self.z_back_color_start_dim : self.z_style_start_dim] # 60-80
                A_z_5 = A_z[:, self.z_style_start_dim :] #80-10
                B_z_1 = B_z[:, 0:self.z_size_start_dim] # 0-20
                B_z_2 = B_z[:, self.z_size_start_dim : self.z_font_color_start_dim] # 20-40
                B_z_3 = B_z[:, self.z_font_color_start_dim : self.z_back_color_start_dim] #40-60
                B_z_4 = B_z[:, self.z_back_color_start_dim : self.z_style_start_dim] # 60-80
                B_z_5 = B_z[:, self.z_style_start_dim :] #80-100
                C_z_1 = C_z[:, 0:self.z_size_start_dim] # 0-20
                C_z_2 = C_z[:, self.z_size_start_dim : self.z_font_color_start_dim] # 20-40
                C_z_3 = C_z[:, self.z_font_color_start_dim : self.z_back_color_start_dim] #40-60
                C_z_4 = C_z[:, self.z_back_color_start_dim : self.z_style_start_dim] # 60-80
                C_z_5 = C_z[:, self.z_style_start_dim :] #80-100
                D_z_1 = D_z[:, 0:self.z_size_start_dim] # 0-20
                D_z_2 = D_z[:, self.z_size_start_dim : self.z_font_color_start_dim] # 20-40
                D_z_3 = D_z[:, self.z_font_color_start_dim : self.z_back_color_start_dim] #40-60
                D_z_4 = D_z[:, self.z_back_color_start_dim : self.z_style_start_dim] # 60-80
                D_z_5 = D_z[:, self.z_style_start_dim :] #80-100
                E_z_1 = E_z[:, 0:self.z_size_start_dim] # 0-20
                E_z_2 = E_z[:, self.z_size_start_dim : self.z_font_color_start_dim] # 20-40
                E_z_3 = E_z[:, self.z_font_color_start_dim : self.z_back_color_start_dim] #40-60
                E_z_4 = E_z[:, self.z_back_color_start_dim : self.z_style_start_dim] # 60-80
                E_z_5 = E_z[:, self.z_style_start_dim :] #80-100
                F_z_1 = F_z[:, 0:self.z_size_start_dim] # 0-20
                F_z_2 = F_z[:, self.z_size_start_dim : self.z_font_color_start_dim] # 20-40
                F_z_3 = F_z[:, self.z_font_color_start_dim : self.z_back_color_start_dim] #40-60
                F_z_4 = F_z[:, self.z_back_color_start_dim : self.z_style_start_dim] # 60-80
                F_z_5 = F_z[:, self.z_style_start_dim :] #80-100

                ## 2. combine with strong supervise
                ''' refer 1: shape, 2: scale, 3: Orientation, 4 Position X, 5 Position Y'''
                # C A same content-1
                A1Co_combine_2C = torch.cat((A_z_1, C_z_2, C_z_3, C_z_4, C_z_5), dim=1)
                mid_A1Co = self.Autoencoder.fc_decoder(A1Co_combine_2C)
                mid_A1Co = mid_A1Co.view(A1Co_combine_2C.shape[0], 256, 4, 4)
                A1Co_2C = self.Autoencoder.decoder(mid_A1Co)

                AoC1_combine_2A = torch.cat((C_z_1, A_z_2, A_z_3, A_z_4, A_z_5), dim=1)
                mid_AoC1 = self.Autoencoder.fc_decoder(AoC1_combine_2A)
                mid_AoC1 = mid_AoC1.view(AoC1_combine_2A.shape[0], 256, 4, 4)
                AoC1_2A = self.Autoencoder.decoder(mid_AoC1)

                # C B same size 2
                B2Co_combine_2C = torch.cat((C_z_1, B_z_2, C_z_3, C_z_4, C_z_5), dim=1)
                mid_B2Co = self.Autoencoder.fc_decoder(B2Co_combine_2C)
                mid_B2Co = mid_B2Co.view(B2Co_combine_2C.shape[0], 256, 4, 4)
                B2Co_2C = self.Autoencoder.decoder(mid_B2Co)

                BoC2_combine_2B = torch.cat((B_z_1, C_z_2, B_z_3, B_z_4, B_z_5), dim=1)
                mid_BoC2 = self.Autoencoder.fc_decoder(BoC2_combine_2B)
                mid_BoC2 = mid_BoC2.view(BoC2_combine_2B.shape[0], 256, 4, 4)
                BoC2_2B = self.Autoencoder.decoder(mid_BoC2)

                # C D same font_color 3
                D3Co_combine_2C = torch.cat((C_z_1, C_z_2, D_z_3, C_z_4, C_z_5), dim=1)
                mid_D3Co = self.Autoencoder.fc_decoder(D3Co_combine_2C)
                mid_D3Co = mid_D3Co.view(D3Co_combine_2C.shape[0], 256, 4, 4)
                D3Co_2C = self.Autoencoder.decoder(mid_D3Co)

                DoC3_combine_2D = torch.cat((D_z_1, D_z_2, C_z_3, D_z_4, D_z_5), dim=1)
                mid_DoC3 = self.Autoencoder.fc_decoder(DoC3_combine_2D)
                mid_DoC3 = mid_DoC3.view(DoC3_combine_2D.shape[0], 256, 4, 4)
                DoC3_2D = self.Autoencoder.decoder(mid_DoC3)

                # C E same back_color 4
                E4Co_combine_2C = torch.cat((C_z_1, C_z_2, C_z_3, E_z_4, C_z_5), dim=1)
                mid_E4Co = self.Autoencoder.fc_decoder(E4Co_combine_2C)
                mid_E4Co = mid_E4Co.view(E4Co_combine_2C.shape[0], 256, 4, 4)
                E4Co_2C = self.Autoencoder.decoder(mid_E4Co)

                EoC4_combine_2E = torch.cat((E_z_1, E_z_2, E_z_3, C_z_4, E_z_5), dim=1)
                mid_EoC4 = self.Autoencoder.fc_decoder(EoC4_combine_2E)
                mid_EoC4 = mid_EoC4.view(EoC4_combine_2E.shape[0], 256, 4, 4)
                EoC4_2E = self.Autoencoder.decoder(mid_EoC4)

                # C F same style 5
                F5Co_combine_2C = torch.cat((C_z_1, C_z_2, C_z_3, C_z_4, F_z_5), dim=1)
                mid_F5Co = self.Autoencoder.fc_decoder(F5Co_combine_2C)
                mid_F5Co = mid_F5Co.view(F5Co_combine_2C.shape[0], 256, 4, 4)
                F5Co_2C = self.Autoencoder.decoder(mid_F5Co)

                FoC5_combine_2F = torch.cat((F_z_1, F_z_2, F_z_3, F_z_4, C_z_5), dim=1)
                mid_FoC5 = self.Autoencoder.fc_decoder(FoC5_combine_2F)
                mid_FoC5 = mid_FoC5.view(FoC5_combine_2F.shape[0], 256, 4, 4)
                FoC5_2F = self.Autoencoder.decoder(mid_FoC5)


                # combine_2C
                A1B2D3E4F5_combine_2C = torch.cat((A_z_1, B_z_2, D_z_3, E_z_4, F_z_5), dim=1)
                mid_A1B2D3E4F5 = self.Autoencoder.fc_decoder(A1B2D3E4F5_combine_2C)
                mid_A1B2D3E4F5 = mid_A1B2D3E4F5.view(A1B2D3E4F5_combine_2C.shape[0], 256, 4, 4)
                A1B2D3E4F5_2C = self.Autoencoder.decoder(mid_A1B2D3E4F5)



                # '''  need unsupervise '''
                A2B3D4E5F1_combine_2N = torch.cat((F_z_1, A_z_2, B_z_3, D_z_4, E_z_5), dim=1)
                mid_A2B3D4E5F1 = self.Autoencoder.fc_decoder(A2B3D4E5F1_combine_2N)
                mid_A2B3D4E5F1 = mid_A2B3D4E5F1.view(A2B3D4E5F1_combine_2N.shape[0], 256, 4, 4)
                A2B3D4E5F1_2N = self.Autoencoder.decoder(mid_A2B3D4E5F1)


                '''
                optimize for autoencoder
                '''

                # 1. recon_loss
                # A_recon_loss = torch.mean(torch.abs(A_img - A_recon))
                A_recon_loss = reconstruction_loss(A_img, A_recon, self.decoder_dist)
                B_recon_loss = reconstruction_loss(B_img, B_recon, self.decoder_dist)
                C_recon_loss = reconstruction_loss(C_img, C_recon, self.decoder_dist)
                D_recon_loss = reconstruction_loss(D_img, D_recon, self.decoder_dist)
                E_recon_loss = reconstruction_loss(E_img, E_recon, self.decoder_dist)
                F_recon_loss = reconstruction_loss(F_img, F_recon, self.decoder_dist)
                recon_loss = A_recon_loss + B_recon_loss + C_recon_loss + D_recon_loss + E_recon_loss + F_recon_loss

                # 2. sup_combine_loss
                A1Co_2C_loss = reconstruction_loss(C_img , A1Co_2C, self.decoder_dist)
                AoC1_2A_loss = reconstruction_loss(A_img , AoC1_2A, self.decoder_dist)
                B2Co_2C_loss = reconstruction_loss(C_img , B2Co_2C, self.decoder_dist)
                BoC2_2B_loss = reconstruction_loss(B_img , BoC2_2B, self.decoder_dist)
                D3Co_2C_loss = reconstruction_loss(C_img , D3Co_2C, self.decoder_dist)
                DoC3_2D_loss = reconstruction_loss(D_img , DoC3_2D, self.decoder_dist)
                E4Co_2C_loss = reconstruction_loss(C_img , E4Co_2C, self.decoder_dist)
                EoC4_2E_loss = reconstruction_loss(E_img , EoC4_2E, self.decoder_dist)
                F5Co_2C_loss = reconstruction_loss(C_img , F5Co_2C, self.decoder_dist)
                FoC5_2F_loss = reconstruction_loss(F_img , FoC5_2F, self.decoder_dist)
                A1B2D3E4F5_2C_loss = reconstruction_loss(C_img , A1B2D3E4F5_2C, self.decoder_dist)
                combine_sup_loss = A1Co_2C_loss + AoC1_2A_loss + B2Co_2C_loss + BoC2_2B_loss + D3Co_2C_loss + DoC3_2D_loss + E4Co_2C_loss + EoC4_2E_loss + F5Co_2C_loss + FoC5_2F_loss + A1B2D3E4F5_2C_loss

                # 3. unsup_combine_loss
                _, A2B3D4E5F1_z = self.Autoencoder(A2B3D4E5F1_2N)
                combine_unsup_loss = reconstruction_loss(F_z_1 , A2B3D4E5F1_z[:, 0:self.z_size_start_dim], self.decoder_dist) \
                                     + reconstruction_loss(A_z_2 , A2B3D4E5F1_z[:, self.z_size_start_dim : self.z_font_color_start_dim], self.decoder_dist) \
                                     + reconstruction_loss(B_z_3 , A2B3D4E5F1_z[:, self.z_font_color_start_dim : self.z_back_color_start_dim], self.decoder_dist) \
                                     + reconstruction_loss(D_z_4 , A2B3D4E5F1_z[:, self.z_back_color_start_dim : self.z_style_start_dim], self.decoder_dist) \
                                     + reconstruction_loss(E_z_5 , A2B3D4E5F1_z[:, self.z_style_start_dim :], self.decoder_dist)

                # whole loss
                vae_unsup_loss = recon_loss + self.lambda_combine * combine_sup_loss + self.lambda_unsup * combine_unsup_loss
                self.auto_optim.zero_grad()
                vae_unsup_loss.backward()
                self.auto_optim.step()

                #　save the log
                f = open(self.log_dir + '/log.txt', 'a')
                f.writelines(['\n', '[{}] recon_loss:{:.3f}  combine_sup_loss:{:.3f}  combine_unsup_loss:{:.3f}'.format(
                        self.global_iter, recon_loss.data, combine_sup_loss.data, combine_unsup_loss.data)])
                f.close()


                if self.viz_on and self.global_iter%self.gather_step == 0:
                    self.gather.insert(iter=self.global_iter,recon_loss=recon_loss.data,
                                    combine_sup_loss=combine_sup_loss.data, combine_unsup_loss=combine_unsup_loss.data)

                if self.global_iter%self.display_step == 0:
                    pbar.write('[{}] recon_loss:{:.3f}  combine_sup_loss:{:.3f}  combine_unsup_loss:{:.3f}'.format(
                        self.global_iter, recon_loss.data, combine_sup_loss.data, combine_unsup_loss.data))

                    if self.viz_on:
                        self.gather.insert(images=A_img.data)
                        self.gather.insert(images=B_img.data)
                        self.gather.insert(images=C_img.data)
                        self.gather.insert(images=D_img.data)
                        self.gather.insert(images=E_img.data)
                        self.gather.insert(images=F_img.data)
                        self.gather.insert(images=F.sigmoid(A_recon).data)
                        self.fonts_viz_reconstruction()
                        self.viz_lines()
                        '''
                        combine show
                        '''
                        self.gather.insert(combine_supimages=F.sigmoid(AoC1_2A).data)
                        self.gather.insert(combine_supimages=F.sigmoid(BoC2_2B).data)
                        self.gather.insert(combine_supimages=F.sigmoid(D3Co_2C).data)
                        self.gather.insert(combine_supimages=F.sigmoid(DoC3_2D).data)
                        self.gather.insert(combine_supimages=F.sigmoid(EoC4_2E).data)
                        self.gather.insert(combine_supimages=F.sigmoid(FoC5_2F).data)
                        self.fonts_viz_combine_recon()

                        self.gather.insert(combine_unsupimages=F.sigmoid(A1B2D3E4F5_2C).data)
                        self.gather.insert(combine_unsupimages=F.sigmoid(A2B3D4E5F1_2N).data)
                        self.fonts_viz_combine_unsuprecon()
                        # self.viz_combine(x)
                        self.gather.flush()
                # Save model checkpoints.
                if self.global_iter%self.save_step == 0:
                    Auto_path = os.path.join(self.model_save_dir, self.viz_name, '{}-Auto.ckpt'.format(self.global_iter))
                    torch.save(self.Autoencoder.state_dict(), Auto_path)
                    print('Saved model checkpoints into {}/{}...'.format(self.model_save_dir, self.viz_name))


                if self.global_iter >= self.max_iter:
                    out = True
                    break

        pbar.write("[Training Finished]")
        pbar.close()
    def test_dsprites(self):
        # self.net_mode(train=True)
        # load pretrained model
        self.restore_model('pretrained')
        for index, sup_package in enumerate(self.data_loader):

            A_img = sup_package['A']
            B_img = sup_package['B']
            D_img = sup_package['D']
            E_img = sup_package['E']
            F_img = sup_package['F']

            A_img = Variable(cuda(A_img, self.use_cuda))
            B_img = Variable(cuda(B_img, self.use_cuda))
            D_img = Variable(cuda(D_img, self.use_cuda))
            E_img = Variable(cuda(E_img, self.use_cuda))
            F_img = Variable(cuda(F_img, self.use_cuda))

            ## 1. A B C seperate(first400: id last600 background)
            A_recon, A_z = self.Autoencoder(A_img)
            B_recon, B_z = self.Autoencoder(B_img)
            D_recon, D_z = self.Autoencoder(D_img)
            E_recon, E_z = self.Autoencoder(E_img)
            F_recon, F_z = self.Autoencoder(F_img)
            ''' refer 1: shape, 2: scale, 3: Orientation, 4 Position X, 5 Position Y'''
            A_z_2 = A_z[:, self.z_size_start_dim: self.z_font_color_start_dim]  # 20-40
            B_z_3 = B_z[:, self.z_font_color_start_dim: self.z_back_color_start_dim]  # 40-60
            D_z_4 = D_z[:, self.z_back_color_start_dim: self.z_style_start_dim]  # 60-80
            E_z_5 = E_z[:, self.z_style_start_dim:]  # 80-100
            F_z_1 = F_z[:, 0:self.z_size_start_dim]  # 0-20

            A2B3D4E5F1_combine_2N = torch.cat((F_z_1, A_z_2, B_z_3, D_z_4, E_z_5), dim=1)
            mid_A2B3D4E5F1 = self.Autoencoder.fc_decoder(A2B3D4E5F1_combine_2N)
            mid_A2B3D4E5F1 = mid_A2B3D4E5F1.view(A2B3D4E5F1_combine_2N.shape[0], 256, 4, 4)
            A2B3D4E5F1_2N = self.Autoencoder.decoder(mid_A2B3D4E5F1)


            # save synthesized image
            self.test_iter = index
            self.gather.insert(test=F.sigmoid(A2B3D4E5F1_2N).data)
            self.fonts_viz_test() # we use fonts vis because it also has 5 attributes
            self.gather.flush()


    def viz_lines(self):
        # self.net_mode(train=False)
        recon_losses = torch.stack(self.gather.data['recon_loss']).cpu()
        combine_sup_loss = torch.stack(self.gather.data['combine_sup_loss']).cpu()
        combine_unsup_loss = torch.stack(self.gather.data['combine_unsup_loss']).cpu()
        iters = torch.Tensor(self.gather.data['iter'])

        legend = []
        for z_j in range(self.z_dim):
            legend.append('z_{}'.format(z_j))
        legend.append('mean')
        legend.append('total')

        if self.win_recon is None:
            self.win_recon = self.viz.line(
                                        X=iters,
                                        Y=recon_losses,
                                        env=self.viz_name+'_lines',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            xlabel='iteration',
                                            title='reconsturction loss',))
        else:
            self.win_recon = self.viz.line(
                                        X=iters,
                                        Y=recon_losses,
                                        env=self.viz_name+'_lines',
                                        win=self.win_recon,
                                        update='append',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            xlabel='iteration',
                                            title='reconsturction loss',))

        if self.win_combine_sup is None:
            self.win_combine_sup = self.viz.line(
                                        X=iters,
                                        Y=combine_sup_loss,
                                        env=self.viz_name+'_lines',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend[:self.z_dim],
                                            xlabel='iteration',
                                            title='combine_sup_loss',))
        else:
            self.win_combine_sup = self.viz.line(
                                        X=iters,
                                        Y=combine_sup_loss,
                                        env=self.viz_name+'_lines',
                                        win=self.win_combine_sup,
                                        update='append',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend[:self.z_dim],
                                            xlabel='iteration',
                                            title='combine_sup_loss',))

        if self.win_combine_unsup is None:
            self.win_combine_unsup = self.viz.line(
                                        X=iters,
                                        Y=combine_unsup_loss,
                                        env=self.viz_name+'_lines',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend[:self.z_dim],
                                            xlabel='iteration',
                                            title='combine_unsup_loss',))
        else:
            self.win_combine_unsup = self.viz.line(
                                        X=iters,
                                        Y=combine_sup_loss,
                                        env=self.viz_name+'_lines',
                                        win=self.win_combine_unsup,
                                        update='append',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend[:self.z_dim],
                                            xlabel='iteration',
                                            title='combine_unsup_loss',))

class DataGather(object):
    def __init__(self):
        self.data = self.get_empty_data_dict()

    def get_empty_data_dict(self):
        return dict(iter=[],
                    recon_loss=[],
                    combine_sup_loss=[],
                    combine_unsup_loss=[],
                    images=[],
                    combine_supimages=[],
                    combine_unsupimages=[],
                    test=[])

    def insert(self, **kwargs):
        for key in kwargs:
            self.data[key].append(kwargs[key])

    def flush(self):
        self.data = self.get_empty_data_dict()

def reconstruction_loss(x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(batch_size)
    elif distribution == 'gaussian':
        x_recon = F.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
    else:
        recon_loss = None

    return recon_loss

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer