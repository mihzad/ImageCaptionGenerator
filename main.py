import math
import os
import torch
import torch.nn as nn

from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW

from torchmetrics.text.bleu import BLEUScore
from torchmetrics.text.bert import BERTScore

from encoder import EncoderCNN
from decoder import DecoderRNN
from dataset import Flickr8kDatasetComposer, flickr_collate_fn

from tester import perform_testing

from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
from datetime import datetime
from zoneinfo import ZoneInfo


img_size = 224
rand_state = 44
embed_size = 256
hidden_size = 512
num_layers = 3
max_len = 40
num_workers = 6
device = torch.device("cuda")#torch.device("cuda" if torch.cuda.is_available() else "cpu")

def curr_time():
    return datetime.now(ZoneInfo('Europe/Kiev'))


def printshare(msg, logfile="training_log.txt"):
    print(msg)

    with open(logfile, "a") as f:
        print(msg, file=f)


def cosannealing_decay_warmup(warmup_steps, T_0, T_mult, decay_factor, base_lr, eta_min):
    # returns the func that performs all the calculations.
    # useful for keeping all the params in one place = scheduler def.
    def lr_lambda(epoch): #0-based epoch
        if epoch < warmup_steps:
            return base_lr * ((epoch + 1) / warmup_steps)

        annealing_step = epoch - warmup_steps

        # calculating which cycle (zero-based) are we in,
        # current cycle length (T_current) and position inside the cycle (t)
        if T_mult == 1:
            cycle = annealing_step // T_0
            t = annealing_step % T_0
            T_current = T_0

        else:
            # fast log-based computation
            cycle = int(math.log((annealing_step * (T_mult - 1)) / T_0 + 1, T_mult))
            sum_steps_of_previous_cycles = T_0 * (T_mult ** cycle - 1) // (T_mult - 1)
            t = annealing_step - sum_steps_of_previous_cycles
            T_current = T_0 * (T_mult ** cycle)


        # enable decay
        eta_max = base_lr * (decay_factor ** cycle)

        # cosine schedule between (eta_min, max_lr]
        lr = eta_min + 0.5 * (eta_max-eta_min) * (1 + math.cos(math.pi * t / T_current))
        return lr/base_lr

    return lr_lambda






def perform_training(encoder, decoder, vocab,
                     training_set,
                     validation_set,
                     epochs, w_decay, batch_size, sub_batch_size,
                     lr, lr_lambda: cosannealing_decay_warmup,
                     pretrained: bool | str = False):

    assert batch_size % sub_batch_size == 0 #screws up gradient accumulation otherwise

    printshare("training preparation...")

    train_loader = DataLoader(training_set, collate_fn=flickr_collate_fn, batch_size=sub_batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(validation_set, collate_fn=flickr_collate_fn, batch_size=sub_batch_size,
                            shuffle=True, num_workers=num_workers)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx["<PAD>"])

    params = list(decoder.parameters()) + list(encoder.fc.parameters()) + list(encoder.bn.parameters())
    optimizer = AdamW(
        params=params, #filter(lambda p: p.requires_grad, params),
        lr=lr, weight_decay=w_decay)

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lr_lambda
    )

    #scheduler = CosineAnnealingLR(
    #    optimizer=optimizer,
    #    T_max=50,
    #    eta_min=1e-8,
    #)

    curr_epoch = 0
    if isinstance(pretrained, str):
        printshare("Loading pretrained model, optimizer & scheduler state dicts...")
        checkpoint = torch.load(pretrained)

        if 'encoder' not in checkpoint:
            printshare("got no encoder, decoder optimizer & scheduler state dicts.")
            return

        else:
            missing_enc, unexpected_enc = encoder.load_state_dict(checkpoint['encoder'], strict=False)
            missing_dec, unexpected_dec = decoder.load_state_dict(checkpoint['decoder'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for g in optimizer.param_groups:
                g['weight_decay'] = w_decay

            #scheduler.load_state_dict(checkpoint["scheduler"])
            scheduler.last_epoch = checkpoint['epoch']
            curr_epoch = checkpoint['epoch'] + 1

            printshare("all the dicts set up successfully.")


        printshare(f"[DEBUG] encoder missing statedict vals: {missing_enc};")
        printshare(f"[DEBUG] encoder unexpected statedict vals: {unexpected_enc}")
        printshare(f"[DEBUG] encoder missing statedict vals: {missing_dec};")
        printshare(f"[DEBUG] encoder unexpected statedict vals: {unexpected_dec}")

    #manual testing cycle
    #while(True):

    #    image, _ = training_set[225]
    #    transform = v2.ToPILImage()
    #    for i in range(16):
    #        img = transform(image[i])
    #        plt.imshow(img)
    #        plt.title(f"Augmented sample #0")
    #        plt.axis('off')
    #        plt.show()

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("checkpoints/stats", exist_ok=True)
    printshare("done.")

    #========== training itself ==========
    while curr_epoch < epochs:
        printshare(f"[{curr_time().strftime('%Y-%m-%d %H:%M:%S')}] epoch {curr_epoch + 1}/{epochs} processing...")

        train_loss, train_bleu = perform_training_epoch(
            encoder=encoder, decoder=decoder, vocab=vocab,
            full_batch_size=batch_size, sub_batch_size=sub_batch_size,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        train_bleu = round(100 * train_bleu, 3)

        printshare(f"training done. bleu: {train_bleu}%")


        printshare(f"[{curr_time().strftime('%Y-%m-%d %H:%M:%S')}] processing validation phase...")
        val_loss, val_bleu = perform_validation_epoch(
            encoder=encoder, decoder=decoder, vocab=vocab,
            val_loader=val_loader,
            criterion=criterion,
        )
        val_bleu = round(100 * val_bleu, 3)
        printshare(f"validation done. bleu: {val_bleu}%\n\n")

        torch.save({ # model
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': curr_epoch,

        }, f'checkpoints/ep_{curr_epoch+1}_tb_{round(train_bleu, 1)}_vb_{round(val_bleu, 1)}_model.pth')

        torch.save({ # stats
            'epoch': curr_epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
        },
            f'checkpoints/stats/ep_{curr_epoch + 1}_tb_{round(train_bleu, 1)}_vb_{round(val_bleu, 1)}_stats.pth')

        curr_epoch += 1

    printshare(f"[{curr_time().strftime('%Y-%m-%d %H:%M:%S')}] training successfully finished.")
    return encoder, decoder


def perform_training_epoch(encoder, decoder, vocab, full_batch_size, sub_batch_size,
                           train_loader, criterion, optimizer, scheduler):

    batch_losses = []
    encoder.train()
    decoder.train()
    bleu_metric = BLEUScore(n_gram=4).to(device)

    accum_steps = math.ceil(full_batch_size / sub_batch_size)  # number of sub-batches per "big batch"

    optimizer.zero_grad()

    for i, (images, captions, ref_list) in enumerate(train_loader):
        images, captions = images.to(device), captions.to(device)
        features = encoder(images)
        outputs = decoder(features, captions[:, :-1])
        #captions are padded, so :-1 does not remove EOS.
        # BUT: criterion.ignore_index = 0 (thats ignore pads) does the job.
        #outputs = outputs[:, :-1, :]  # remove last time step=EOS
        loss = criterion(outputs.reshape(-1, len(vocab)),
                         captions[:, 1:].reshape(-1))

        loss = loss / accum_steps # smoothing the magnitude for accumulating
        loss.backward()
        batch_losses.append(loss.item())

        preds = outputs.argmax(dim=-1)
        pred_sentences = vocab.detokenize(preds)

        if len(pred_sentences) > 0:
            bleu_metric.update(pred_sentences, ref_list)

        if (i + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    scheduler.step()
    epoch_loss = sum(batch_losses) / len(batch_losses)
    bleu_avg = bleu_metric.compute().item()
    return epoch_loss, bleu_avg


def perform_validation_epoch(encoder, decoder, vocab, val_loader, criterion):
    encoder.eval()
    decoder.eval()
    bleu_metric = BLEUScore(n_gram=4).to(device)
    with torch.no_grad():
        batch_losses = []

        for images, captions, ref_list in val_loader:
            images, captions = images.to(device), captions.to(device)

            features = encoder(images)

            hidden = None

            input_tokens_batch = torch.full(size=(val_loader.batch_size,),
                                            fill_value=vocab.word2idx["<SOS>"]).unsqueeze(1).to(device) #<sos> input
            generated_outputs = []

            for step in range(1, max_len):
                # On first step, include image features
                output, hidden = decoder.forward_inference_step(input_tokens_batch, hidden,
                                                                features_emb=features if step == 1 else None)
                # Choose the most probable next token (greedy decoding)
                predicted_batch = output.argmax(-1)
                # Feed predicted batch into next step
                input_tokens_batch = predicted_batch
                generated_outputs.append(output)

            batch_output = torch.stack(generated_outputs, dim=1)
            batch_losses.append(criterion(batch_output.reshape(-1, len(vocab)),
                                          captions[:, 1:].reshape(-1))
                                .item())

            generated_captions = vocab.detokenize(batch_output.argmax(-1))
            if len(generated_captions) > 0:
                bleu_metric.update(generated_captions, ref_list)

        epoch_loss = sum(batch_losses) / len(batch_losses)
        bleu_avg = bleu_metric.compute().item()
        return epoch_loss, bleu_avg





def custom_loader(path):
    return Image.open(path, formats=["JPEG"])




if __name__ == '__main__':

    train_transform = v2.Compose([
        v2.RandomResizedCrop(
            size=(img_size, img_size),
            scale=(0.8, 1.0),
            ratio=(7.0 / 8.0, 8.0 / 7.0)
        ),

        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    nontrain_transform = v2.Compose([
        v2.Resize(size=(img_size, img_size)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.485, 0.456, 0.406),
                     (0.229, 0.224, 0.225))
    ])


    print("Loading dataset...")
    dataset = Flickr8kDatasetComposer(root_dir="data", max_len=max_len, vocab_threshold=3)

    # Create subsets
    train_set = dataset.get_subset("train", transform=train_transform)
    val_set = dataset.get_subset("val", transform=nontrain_transform)
    test_set = dataset.get_subset("test", transform=nontrain_transform)
    vocabulary = dataset.vocab

    print(f"Vocab size: {len(vocabulary)}")
    print(f"Train captions: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, len(vocabulary), num_layers, dropout=0.3).to(device)



    #perform_training(encoder=encoder,decoder=decoder, vocab=vocabulary,
    #                 training_set=train_set, validation_set=val_set,
    #                 epochs=600, w_decay=1e-4, batch_size=64, sub_batch_size=64,
    #                 lr=1e-3, lr_lambda=cosannealing_decay_warmup(
    #                   warmup_steps=5, T_0=10, T_mult=1.2, decay_factor=0.9, base_lr=1e-3, eta_min=1e-8),
    #                 pretrained=False)

    perform_testing(encoder=encoder, decoder=decoder, vocab=vocabulary, batch_size=64, testing_set=test_set,
                    weights_file='checkpoints/models_e256_h512_l3/ep_39_tb_16.4_vb_15.1_model.pth')
