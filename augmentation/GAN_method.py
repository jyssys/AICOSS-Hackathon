import torch
from torchvision.utils import save_image
import torch.nn.functional as F

from GAN_data_loader import data_loader

import pandas as pd
import os
import logging

def train_model(discriminator, generator, criterion, g_optimizer, d_optimizer, device, num_epoch, batch_size, dir_name):
    df = pd.DataFrame(columns=['image_path'] + [f'class_{i}' for i in range(60)])
    
    for epoch in range(num_epoch):
        for i, data in enumerate(data_loader):
            current_batch_size = data['image'].size(0)
            
            # make ground truth (labels) -> 1 for real, 0 for fake
            real_label = torch.full((current_batch_size, 1), 1, dtype=torch.float32).to(device)
            fake_label = torch.full((current_batch_size, 1), 0, dtype=torch.float32).to(device)

            inputs = data['image'].to(device)
            label = data['labels'].to(device)

            # reshape real images from dataset
            real_images = inputs.reshape(current_batch_size, -1).to(device)

            """
            FOR CONDITIONAL GAN
            """
            # Encode data label's with 'one hot encoding'
            if len(label.shape) > 1 and label.shape[1] > 1:
                label = label.argmax(dim=1)
            
            label_encoded = F.one_hot(label, num_classes=60).to(device)
            
            # concat real images with 'label encoded vector'
            real_images_concat = torch.cat((real_images, label_encoded), 1)
    
            # +---------------------+
            # |   train Generator   |
            # +---------------------+

            print(f'Epoch : {epoch + 1}, {i}th batch train start!')
            
            # Initialize grad
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()

            # make fake images with generator & noise vector 'z'
            z = torch.randn(current_batch_size, 100).to(device)

            """
            FOR CONDITIONAL GAN
            """
            # concat noise vector z with encoded labels
            z_concat = torch.cat((z, label_encoded), 1)
            fake_images = generator(z_concat)
            fake_images_concat = torch.cat((fake_images, label_encoded), 1)

            # Compare result of discriminator with fake images & real labels
            # If generator deceives discriminator, g_loss will decrease
            g_loss = criterion(discriminator(fake_images_concat), real_label)

            # Train generator with backpropagation
            g_loss.backward()
            g_optimizer.step()

            # +---------------------+
            # | train Discriminator |
            # +---------------------+

            # Initialize grad
            d_optimizer.zero_grad()
            g_optimizer.zero_grad()

            # make fake images with generator & noise vector 'z'
            z = torch.randn(current_batch_size, 100).to(device)

            """
            FOR CONDITIONAL GAN
            """
            # concat noise vector z with encoded labels
            z_concat = torch.cat((z, label_encoded), 1)
            fake_images = generator(z_concat)
            fake_images_concat = torch.cat((fake_images, label_encoded), 1)
        
            # Calculate fake & real loss with generated images above & real images
            fake_loss = criterion(discriminator(fake_images_concat), fake_label)
            real_loss = criterion(discriminator(real_images_concat), real_label)
            d_loss = (fake_loss + real_loss) / 2

            # Train discriminator with backpropagation
            # In this part, we don't train generator
            d_loss.backward()
            d_optimizer.step()

            d_performance = discriminator(real_images_concat).mean()
            g_performance = discriminator(fake_images_concat).mean()

            if i % 100 == 99:
                print("Epoch [ {}/{} ]  Step [ {}/{} ]  d_loss : {:.5f}  g_loss : {:.5f}".format(epoch, num_epoch, i+1, len(data_loader), d_loss.item(), g_loss.item()))
                logging.info("Epoch [ {}/{} ]  Step [ {}/{} ]  d_loss : {:.5f}  g_loss : {:.5f}".format(epoch, num_epoch, i+1, len(data_loader), d_loss.item(), g_loss.item()))
    
        # print discriminator & generator's performance
        print("Epoch {} discriminator performance : {:.2f}  generator performance : {:.2f}".format(epoch + 1, d_performance, g_performance))
        logging.info("Epoch {} discriminator performance : {:.2f}  generator performance : {:.2f}".format(epoch + 1, d_performance, g_performance))
        
        # Save fake images in each epoch
        samples = fake_images.reshape(current_batch_size, 3, 188, 188)
        save_image(samples, os.path.join(dir_name, 'CGAN_fake_samples{}.png'.format(epoch + 1)))
        
        for i in range(current_batch_size):
            image_path = f'{dir_name}/CGAN_fake_image_{epoch}_{i}.png'

            image_to_save = fake_images[i].view(3, 188, 188)

            save_image(image_to_save, image_path)

            label = label_encoded[i].argmax().item()  # one-hot 인코딩된 레이블을 정수로 변환
            df.loc[len(df)] = [image_path] + [int(j == label) for j in range(60)]  # 데이터프레임에 추가

    # 학습 루프 종료 후 CSV 파일로 저장
    df.to_csv(f'{dir_name}/generated_images_labels.csv', index=False)