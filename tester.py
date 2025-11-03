import torch
from torch.utils.data import DataLoader
from dataset import flickr_collate_fn
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

from PIL import Image
from torchmetrics.text.bleu import BLEUScore
from torchmetrics.text.bert import BERTScore

device = torch.device("cuda")#torch.device("cuda" if torch.cuda.is_available() else "cpu")



def perform_testing(encoder, decoder, vocab, batch_size, testing_set, weights_file: str):
    def printshare(msg, logfile=f"{weights_file}_test.txt"):
        print(msg)

        with open(logfile, "a") as f:
            print(msg, file=f)

    printshare("performing testing...")
    testing_loader = DataLoader(testing_set, batch_size=batch_size, shuffle=True, collate_fn=flickr_collate_fn, num_workers=6)

    checkpoint = torch.load(weights_file)
    encoder.load_state_dict(checkpoint['encoder'], strict=False)
    decoder.load_state_dict(checkpoint['decoder'], strict=False)
    encoder.eval()
    decoder.eval()
    bleu_metric = BLEUScore(n_gram=4).to(device)
    bert_metric = BERTScore(model_name_or_path='microsoft/deberta-base-mnli', device=device)
    with torch.no_grad():
        bert_scores = []
        for images, captions, ref_list in testing_loader:
            images, captions = images.to(device), captions.to(device)

            features = encoder(images)

            hidden = None

            input_tokens_batch = torch.full(size=(batch_size,),
                                            fill_value=vocab.word2idx["<SOS>"]).unsqueeze(1).to(device)  # <sos> input
            generated_outputs = []

            for step in range(1, testing_set.max_len):
                # On first step, include image features
                output, hidden = decoder.forward_inference_step(input_tokens_batch, hidden,
                                                                features_emb=features if step == 1 else None)
                # Choose the most probable next token (greedy decoding)
                predicted_batch = output.argmax(-1)
                # Feed predicted batch into next step
                input_tokens_batch = predicted_batch
                generated_outputs.append(output)

            batch_output = torch.stack(generated_outputs, dim=1)

            generated_captions = vocab.detokenize(batch_output.argmax(-1))
            if len(generated_captions) > 0:
                bleu_metric.update(generated_captions, ref_list)
                gc_list = [gc for gc in generated_captions for _ in range(len(ref_list[0]))]
                rf_flattened = [r for refs in ref_list for r in refs]
                bert_scores.append(bert_metric(gc_list, rf_flattened))

        bleu_avg = bleu_metric.compute().item()

        all_f1 = torch.cat([d['f1'] for d in bert_scores])
        all_precision = torch.cat([d['precision'] for d in bert_scores])
        all_recall = torch.cat([d['recall'] for d in bert_scores])
        bert_avg = {
            'f1': all_f1.mean().item(),
            'precision': all_precision.mean().item(),
            'recall': all_recall.mean().item()
        }
        printshare(f"bleu: {bleu_avg}; bert: {bert_avg}")

def perform_manual_testing(encoder, decoder, vocab, img_transform, weights_file: str):
    own_set = []

    img_names = ["dogs_playing.jpg", "people_traveling.jpg", "people_cycling.jpg"]
    for img_name in img_names:
        img_path = f"own_set/{img_name}"
        img = Image.open(img_path, formats=["JPEG"]).convert("RGB")
        img_tensor = img_transform(F.to_tensor(img))
        img_ready = img_transform(img_tensor)
        own_set.append(img_ready)

    checkpoint = torch.load(weights_file)
    encoder.load_state_dict(checkpoint['encoder'], strict=False)
    decoder.load_state_dict(checkpoint['decoder'], strict=False)
    encoder.eval()
    decoder.eval()
    with (torch.no_grad()):

        images = torch.stack(own_set).to(device)

        features = encoder(images)

        hidden = None
        input_tokens_batch = torch.full(size=(len(own_set),),
                                        fill_value=vocab.word2idx["<SOS>"]).unsqueeze(1).to(device)  # <sos> input
        generated_outputs = []

        for step in range(1, 40):
            # On first step, include image features
            output, hidden = decoder.forward_inference_step(input_tokens_batch, hidden,
                                                            features_emb=features if step == 1 else None)
            # Choose the most probable next token (greedy decoding)
            predicted_batch = output.argmax(-1)
            # Feed predicted batch into next step
            input_tokens_batch = predicted_batch
            generated_outputs.append(output)

        batch_output = torch.stack(generated_outputs, dim=1)

        generated_captions = vocab.detokenize(batch_output.argmax(-1))

        for i, c in zip(img_names, generated_captions):
            print(f"for image {i} gen caption: \"{c}\"")