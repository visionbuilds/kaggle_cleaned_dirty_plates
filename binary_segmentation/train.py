import time
import torch.nn as nn
import numpy as np
import segmentation_models_pytorch as smp
import torchvision
import pickle
import os
import torch
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
import pandas as pd
from datetime import timedelta
from segmentation_models_pytorch.utils.losses import DiceLoss,CrossEntropyLoss,BCEWithLogitsLoss
ENCODER = "resnet34"
EPOCHS = 30
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
def train_segmentation_model(root,BATCH_SIZE,class_,model):
    root_model = os.path.join(root, 'models')
    os.makedirs(root_model, exist_ok=True)
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER)
    with open(root+'/train_dataset.pkl', 'rb') as train_dataset_file: # Open pickle file with train dataset
      train_dataset = pickle.load(train_dataset_file)
      train_dataset_file.close()
    #

    with open(root+'/valid_dataset.pkl', 'rb') as valid_dataset_file: # Open pickle file with valid dataset
      valid_dataset = pickle.load(valid_dataset_file)
      valid_dataset_file.close()

    print('len(train_dataset) ',len(train_dataset))
    print('len(valid_dataset) ',len(valid_dataset))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,drop_last=True)  #, num_workers=12)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False,drop_last=True)  #, num_workers=12)
    loss = DiceLoss()  # for binary segmentation

    metrics = [
        smp.utils.metrics.IoU(threshold=0.5, activation=None, ignore_channels=None),
        smp.utils.metrics.Fscore(threshold=0.5, activation=None, ignore_channels=None),
        smp.utils.metrics.Accuracy(threshold=0.5, activation=None, ignore_channels=None),
        smp.utils.metrics.Recall(threshold=0.5, activation=None, ignore_channels=None),
        smp.utils.metrics.Precision(threshold=0.5, activation=None, ignore_channels=None)
               ]
    optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=LR)])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    CyclicLR_activated=False

    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True)

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True)

    iou_train_hist = []
    iou_valid_hist = []
    loss_train_hist = []
    loss_valid_hist = []
    max_score = 0 # max value of IoU score
    max_fscore=0
    max_accuracy=0
    max_recall=0
    max_precision=0
    min_score = 100 # min value of loss
    total_time=0
    # train model for 40 epochs
    df_name=root+ '/' + class_ + model.name +'train_len_'+str(len(train_dataset))+'_valid_len_'+str(len(valid_dataset))+'_bs' + \
            str(BATCH_SIZE)+'_re_scale' + '_training_logs.csv'

    try:
        df = pd.read_csv(df_name)
        print(df)
        max_score = df['val_IOU'].max()
        max_fscore = df['val_fscore'].max()
        max_accuracy = df['val_accuracy'].max()
        max_recall = df['val_recall'].max()
        max_precision = df['val_precision'].max()
        min_score = df['val_loss'].min()
        best_epoch_score = df['val_IOU'].astype(float).idxmax()
        best_epoch_fscore = df['val_fscore'].astype(float).idxmax()
        best_epoch_accuracy = df['val_accuracy'].astype(float).idxmax()
        best_epoch_recall = df['val_recall'].astype(float).idxmax()
        best_epoch_precision = df['val_precision'].astype(float).idxmax()
        best_epoch_loss = df['val_loss'].astype(float).idxmin()
        total_time = df.training_time.sum().round()
        print(f"max IOU {max_score} is on {best_epoch_score} epoch, min LOSS {min_score} is on {best_epoch_loss} epoch ")
        print(
            f"max Fscore {max_fscore} is on {best_epoch_fscore} epoch, max Accuracy {max_accuracy} is on {best_epoch_accuracy} epoch, max Precision {max_precision} is on {best_epoch_precision} epoch, max recall {max_recall} is on {best_epoch_recall} epoch ")

    except:
        df = pd.DataFrame(
            columns=['epoch', 'train_loss', 'train_IOU', 'train_fscore', 'train_accuracy', 'train_precision',
                     'train_recall', 'val_loss', 'val_IOU', 'val_fscore', 'val_accuracy', 'val_precision', 'val_recall',
                     'learning_rate', 'training_time'])
        df.to_csv(df_name, index=False)
        best_epoch_score = best_epoch_loss = best_epoch_fscore = best_epoch_accuracy = best_epoch_recall = best_epoch_precision = 0

    for epoch in range(0, EPOCHS):
        start_time = time.time()
        torch.cuda.empty_cache()  # Очистим кеш памяти
        print('\nEpoch: {}'.format(epoch))
        torch.save(model, root_model + '/_model.pth')
        # gpu_usage()  # Посмотрим сколько памяти требуется
        start_time = time.time()
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        loss_name = [i for i in train_logs.keys() if 'loss' in i][0]
        finish_time = time.time()
        training_time = finish_time - start_time
        total_time += training_time
        iou_train_hist.append(train_logs['iou_score'])
        iou_valid_hist.append(valid_logs['iou_score'])
        print('train_logs', train_logs)
        print('train_logs', valid_logs)
        loss_train_hist.append(train_logs[loss_name])
        loss_valid_hist.append(valid_logs[loss_name])



        # gpu_usage() # Посмотрим сколько памяти требуется
        # scheduler.step(valid_logs['iou_score'])
        # scheduler.step()

        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, root_model + '/_best_model_iou.pth')
            best_epoch_score = epoch
            print('Model Best IOU saved!')

        if min_score > valid_logs[loss_name]:
            min_score = valid_logs[loss_name]
            torch.save(model, root_model + '/_best_model_loss.pth')
            best_epoch_loss = epoch
            print('Model Best Loss saved!')

        if max_fscore < valid_logs['fscore']:
            max_fscore = valid_logs['fscore']
            torch.save(model, root_model + '/_best_model_fscore.pth')
            best_epoch_fscore = epoch
            print('Model Best fscore saved!')

        if max_accuracy < valid_logs['accuracy']:
            max_accuracy = valid_logs['accuracy']
            torch.save(model, root_model + '/_best_model_accuracy.pth')
            best_epoch_accuracy = epoch
            print('Model Best accuracy saved!')

        if max_precision < valid_logs['precision']:
            max_precision = valid_logs['precision']
            torch.save(model, root_model + '/_best_model_precision.pth')
            best_epoch_precision = epoch
            print('Model Best precision saved!')

        if max_recall < valid_logs['recall']:
            max_recall = valid_logs['recall']
            torch.save(model, root_model + '/_best_model_recall.pth')
            best_epoch_recall = epoch
            print('Model Best recall saved!')

        # torch.save(model,root_model + '/'+'epoch_'+str(epoch)+'_loss_'+str(round(float(valid_logs['dice_loss']),2))+ '_model.pth')
        for file in os.listdir(
                root_model):  # Удаляем все существующие изображения с графиками из папки root_model, чтобы там всегда был сохранён только последний график
            if file.endswith('png'):
                os.remove(os.path.join(root_model, file))

        print(
            f"max IOU {max_score} is on {best_epoch_score} epoch, min LOSS {min_score} is on {best_epoch_loss} epoch ")
        print(
            f"max Fscore {max_fscore} is on {best_epoch_fscore} epoch, max Accuracy {max_accuracy} is on {best_epoch_accuracy} epoch, max Precision {max_precision} is on {best_epoch_precision} epoch, max recall {max_recall} is on {best_epoch_recall} epoch ")
        print(f'{training_time=}')
        print(f'total_time= {timedelta(seconds=total_time)}')
        data = {
            'epoch': epoch,
            'train_loss': train_logs[loss_name],
            'train_IOU': train_logs['iou_score'],
            'train_fscore': train_logs['fscore'],
            'train_accuracy': train_logs['accuracy'],
            'train_precision': train_logs['precision'],
            'train_recall': train_logs['recall'],
            'val_loss': valid_logs[loss_name],
            'val_IOU': valid_logs['iou_score'],
            'val_fscore': valid_logs['fscore'],
            'val_accuracy': valid_logs['accuracy'],
            'val_precision': valid_logs['precision'],
            'val_recall': valid_logs['recall'],
            'learning_rate': optimizer.param_groups[0]['lr'],
            'training_time': training_time
        }
        # print(pd.DataFrame(data=[[*data.values()]],columns=df.columns))
        df = pd.concat([df, pd.DataFrame(data=[[*data.values()]], columns=df.columns, index=[data['epoch']])])
        # print(df)
        df.to_csv(df_name, index=False)

        markersize = 8
        fontsize = 12
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, _, _)) = plt.subplots(3, 3, figsize=(12, 4))
        fig.suptitle(f'{class_}_' + model.name + '_: encoder={ENCODER},\nloss=DiceLoss, '
                                                 f'lr={optimizer.param_groups[0]["lr"]}, bs={BATCH_SIZE}')
        # fig.suptitle(f'{str(CLASS)}_'+model.name+'_: encoder={ENCODER},\nloss=DiceLoss, '
        #              f'lr={optimizer.param_groups[0]["lr"]}, bs={BATCH_SIZE}, re_scale={str(COEF_MIN)}')
        ax1.plot(df['train_IOU'], label='iou_score_train')
        ax1.plot(df['val_IOU'], label='iou_score_valid')
        ax1.plot(best_epoch_score, df.loc[best_epoch_score, 'val_IOU'], marker="o", markersize=markersize,
                 markeredgecolor="green", markerfacecolor="green")
        text = str(round(df.loc[best_epoch_score, 'val_IOU'], 2)) + '(epoch ' + str(best_epoch_score) + ')'
        ax1.annotate(text, (best_epoch_score + 5, df.loc[best_epoch_score, 'val_IOU'] - 0.03), fontsize=fontsize)
        ax1.legend()
        ax1.grid()
        # ax2.plot(loss_train_hist, label='DiceLoss_train')
        # ax2.plot(loss_valid_hist, label='DiceLoss_valid')
        ax2.plot(df['train_loss'], label=loss_name + '_train')
        ax2.plot(df['val_loss'], label=loss_name + '_valid')
        ax2.plot(best_epoch_loss, df.loc[best_epoch_loss, 'val_loss'], marker="o", markersize=markersize,
                 markeredgecolor="green", markerfacecolor="green")
        text = str(round(df.loc[best_epoch_loss, 'val_loss'], 2)) + '(epoch ' + str(best_epoch_loss) + ')'
        ax2.annotate(text, (best_epoch_loss + 5, df.loc[best_epoch_loss, 'val_loss'] - 0.03), fontsize=fontsize)
        ax2.legend()
        ax2.grid()
        # plt.show()
        ax3.plot(df['train_fscore'], label='fscore' + '_train')
        ax3.plot(df['val_fscore'], label='fscore' + '_valid')
        ax3.plot(best_epoch_fscore, df.loc[best_epoch_fscore, 'val_fscore'], marker="o", markersize=markersize,
                 markeredgecolor="green", markerfacecolor="green")
        text = str(round(df.loc[best_epoch_fscore, 'val_fscore'], 2)) + '(epoch ' + str(best_epoch_fscore) + ')'
        ax3.annotate(text, (best_epoch_fscore + 5, df.loc[best_epoch_fscore, 'val_fscore'] - 0.03), fontsize=fontsize)
        ax3.legend()
        ax3.grid()

        ax4.plot(df['train_accuracy'], label='accuracy' + '_train')
        ax4.plot(df['val_accuracy'], label='accuracy' + '_valid')
        ax4.plot(best_epoch_accuracy, df.loc[best_epoch_loss, 'val_accuracy'], marker="o", markersize=markersize,
                 markeredgecolor="green", markerfacecolor="green")
        text = str(round(df.loc[best_epoch_accuracy, 'val_accuracy'], 2)) + '(epoch ' + str(best_epoch_accuracy) + ')'
        ax4.annotate(text, (best_epoch_accuracy + 5, df.loc[best_epoch_accuracy, 'val_accuracy'] - 0.03),
                     fontsize=fontsize)
        ax4.legend()
        ax4.grid()

        ax5.plot(df['train_precision'], label='precision' + '_train')
        ax5.plot(df['val_precision'], label='precision' + '_valid')
        ax5.plot(best_epoch_precision, df.loc[best_epoch_precision, 'val_precision'], marker="o", markersize=markersize,
                 markeredgecolor="green", markerfacecolor="green")
        text = str(round(df.loc[best_epoch_precision, 'val_precision'], 2)) + '(epoch ' + str(
            best_epoch_precision) + ')'
        ax5.annotate(text, (best_epoch_precision + 5, df.loc[best_epoch_precision, 'val_precision'] - 0.03),
                     fontsize=fontsize)
        ax5.legend()
        ax5.grid()

        ax6.plot(df['train_recall'], label='recall' + '_train')
        ax6.plot(df['val_recall'], label='recall' + '_valid')
        ax6.plot(best_epoch_recall, df.loc[best_epoch_recall, 'val_recall'], marker="o", markersize=markersize,
                 markeredgecolor="green", markerfacecolor="green")
        text = str(round(df.loc[best_epoch_recall, 'val_recall'], 2)) + '(epoch ' + str(best_epoch_recall) + ')'
        ax6.annotate(text, (best_epoch_recall + 5, df.loc[best_epoch_recall, 'val_recall'] - 0.03), fontsize=fontsize)
        ax6.legend()
        ax6.grid()

        ax7.plot(df['learning_rate'], label='learning_rate')
        ax7.grid()

        fig.savefig(root_model + '/' + class_ + model.name + ENCODER + '_' + loss_name + '_' + str(
            round(float(min_score), 2)) + '_lr'
                    + str(optimizer.param_groups[0]['lr']) + '_bs' + str(BATCH_SIZE)
                    + '_re_scale' + '.png')
        fig.clf()

if __name__ =='__main__':
    root=r'A:\pycharm_projects\plates\data\dataset'
    BATCH_SIZE = 6
    CLASS = 'plate'
    model = smp.PAN(
        encoder_name=ENCODER,  # choose encoder, e.g. 'resnet18'
        encoder_weights='imagenet',
        # encoder_weights=None,
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,  # model output classes (number of classes in your dataset)
        activation='sigmoid')
    train_segmentation_model(root=root, BATCH_SIZE=BATCH_SIZE, class_=CLASS, model=model)