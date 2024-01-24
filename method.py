import torch
from torch.nn import Sigmoid
from utils import mixup
from data_loader import get_transforms, get_data_loaders
from config import configs

import numpy as np

import warnings
warnings.filterwarnings(action='ignore')
import time
import logging

def create_data_loaders_by_num(num):
    config = next((item for item in configs if item['num'] == num), None)
    if config is not None:
        train_transform, test_transform = get_transforms(config['transform_resize'], config['is_crop'])
        train_loader, val_loader, test_loader = get_data_loaders(config['batch_size'], train_transform, test_transform)

        return train_loader, val_loader, test_loader

    else:
        raise ValueError(f"No configuration found for 'num'={num}")

def train_model(model, criterion, optimizer, scheduler, device, num, num_epochs = 15, mode = 'sam_normal'):
    train_loader, val_loader, test_loader = create_data_loaders_by_num(num)
    
    if mode == 'sam_normal':
        best_val_loss = float('inf')
        best_model = None

        for epoch in range(num_epochs):
            model.train()  # model training mode
            running_loss = 0.0
            train_loss = []
            
            start_time = time.time()  # 에폭 시작 시간 기록

            for i, data in enumerate(train_loader, 0):
                inputs = data['image'].to(device)
                labels = data['labels'].to(device)

                print(f'Epoch : {epoch + 1}, {i}th batch train start!')
                
                optimizer.zero_grad()

                # First Forward step
                outputs1 = model(inputs)
                loss1 = criterion(outputs1, labels)
                loss1.backward()
                optimizer.first_step(zero_grad = True)
                
                # second real update with backward
                outputs2 = model(inputs)
                criterion(outputs2, labels).backward()
                optimizer.second_step(zero_grad=True)
                loss2 = criterion(outputs2, labels)

                running_loss += loss2.item()
                train_loss.append(loss2.item())
                
                if i % 100 == 99:  # 매 100 배치마다 로스 출력
                    print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100:.4f}')
                    logging.info(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100:.4f}')
                    running_loss = 0.0

            _val_loss = val_model(model, criterion, val_loader, device)
            _train_loss = np.mean(train_loss)
            
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            end_time = time.time()
            total_time = end_time - start_time
            
            print(f'Epoch {epoch + 1} completed, Train Loss : [{_train_loss:.4f}], Val Loss : [{_val_loss:.4f}], Current lr : [{current_lr:.6f}], Total time: {total_time:.2f} seconds')
            logging.info(f'Epoch {epoch + 1} completed, Train Loss : [{_train_loss:.4f}], Val Loss : [{_val_loss:.4f}],  Current lr : [{current_lr:.6f}], Total time: {total_time:.2f} seconds')
            
            if best_val_loss > _val_loss:
                best_val_loss = _val_loss
                best_model = model
        
        print(f'train complete!')
        return best_model
    
    elif mode == 'sam_mix_up':
        best_val_loss = float('inf') 
        best_model = None

        for epoch in range(num_epochs):
            model.train()  # model training mode
            running_loss = 0.0
            train_loss = []
            
            start_time = time.time()  # 에폭 시작 시간 기록

            for i, data in enumerate(train_loader, 0):
                inputs = data['image'].to(device)
                labels = data['labels'].to(device)

                # apply mixup
                mix_decision = np.random.rand()
                
                if mix_decision >= 0.: 
                    inputs, targets = mixup(inputs, labels, alpha = 1)
                    lam = targets[2]                
                    
                print(f'Epoch : {epoch + 1}, {i}th batch train start!')
                optimizer.zero_grad()
                
                if mix_decision >= 0.:
                    # First Forward step
                    outputs1 = model(inputs)
                    loss1 = lam * criterion(outputs1, targets[0]) + (1 - lam) * criterion(outputs1, targets[1])
                    loss1.backward()
                    optimizer.first_step(zero_grad = True)
                    
                    # second real update with backward
                    outputs2 = model(inputs)
                    criterion(outputs2, labels).backward()
                    optimizer.second_step(zero_grad=True)
                    loss2 = lam * criterion(outputs2, targets[0]) + (1 - lam) * criterion(outputs2, targets[1])
                else:
                    # First Forward step
                    outputs1 = model(inputs)
                    loss1 = criterion(outputs1, labels)
                    loss1.backward()
                    optimizer.first_step(zero_grad = True)
                    
                    # second real update with backward
                    outputs2 = model(inputs)
                    criterion(outputs2, labels).backward()
                    optimizer.second_step(zero_grad=True)
                    loss2 = criterion(outputs2, labels)

                running_loss += loss2.item()
                train_loss.append(loss2.item())

                if i % 100 == 99:  # 매 100 배치마다 로스 출력
                    print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100:.4f}')
                    logging.info(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100:.4f}')
                    running_loss = 0.0

            _val_loss = val_model(model, criterion, val_loader, device)
            _train_loss = np.mean(train_loss)
            
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            end_time = time.time()
            total_time = end_time - start_time
            
            print(f'Epoch {epoch + 1} completed, Train Loss : [{_train_loss:.4f}], Val Loss : [{_val_loss:.4f}], Current lr : [{current_lr:.6f}], Total time: {total_time:.2f} seconds')
            logging.info(f'Epoch {epoch + 1} completed, Train Loss : [{_train_loss:.4f}], Val Loss : [{_val_loss:.4f}],  Current lr : [{current_lr:.6f}], Total time: {total_time:.2f} seconds')
            
            if best_val_loss > _val_loss:
                best_val_loss = _val_loss
                best_model = model
        
        print(f'train complete!')
        return best_model


def val_model(model, criterion, val_loader, device):
    model.eval()
    val_loss = []
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            inputs = data['image'].to(device)
            labels = data['labels'].to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss.append(loss.item())
        
        _val_loss = np.mean(val_loss)
    
    return _val_loss


def predict(model, device, num):
    train_loader, val_loader, test_loader = create_data_loaders_by_num(num)
        
    model.eval()  # model eval mode
    sigmoid = Sigmoid()
    predictions = []

    print(f'test pred start!')
    
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs = data['image'].to(device)
            outputs = model(inputs)
            outputs = sigmoid(outputs)
            predictions.extend(outputs.cpu().numpy())

    print(f'test pred complete!')

    return predictions