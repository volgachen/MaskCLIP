import torch
import clip
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Extract and save the CLIP visual weights')
    parser.add_argument('--model', default='RN50', choices=['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT32', 'ViT16', 'RN101_blur', 'RN101_noise'], help='clip model name')
    parser.add_argument('--backbone', action='store_true', help='Prepend the word backbone to the key so that it can be directly loaded as a checkpoint')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    name_mapping = {'RN50': 'RN50', 'RN101': 'RN101', 'RN50x4': 'RN50x4', \
        'RN50x16': 'RN50x16', 'ViT32': 'ViT-B/32', 'ViT16': 'ViT-B/16'}
    clip_model, preprocess = clip.load(name_mapping[args.model], device='cpu')
    state_dict = clip_model.state_dict()

    result_model = {'meta': {}, 'state_dict': {}}
    all_model = dict()
    stem_mapping = {'conv1': 0, 'bn1': 1, 'conv2': 3, 'bn2': 4, 'conv3': 6, 'bn3':7}
    clip_keys = []
    prefix = 'visual'
    for key in state_dict.keys():
        if prefix in key:
            if 'attnpool' in key:
                if 'proj' in key:
                    proj_name = key.split('.')[2]
                    weight_name = key.split('.')[3]
                    if proj_name not in all_model:
                        all_model[proj_name] = {}
                    all_model[proj_name][weight_name] = state_dict[key].float()
            else:
                new_key = key[len(f'{prefix}.'):]
                if 'layer' not in new_key:
                    layer_name, layer_type = new_key.split('.')
                    new_key = 'stem.{}.{}'.format(stem_mapping[layer_name], layer_type)
                if 'downsample' in new_key:
                    splits = new_key.split('.')
                    new_key = '{}.{}.{}.{}.{}'.format(splits[0], splits[1], splits[2], \
                        int(splits[3])+1, splits[4])
                if args.backbone:
                    new_key = 'backbone.' + new_key
                clip_keys.append(new_key)
                result_model['state_dict'].update({new_key: state_dict[key].float()})

    if args.backbone:
        torch.save(result_model, f'pretrain/{args.model}_clip_backbone.pth')
    else:
        torch.save(result_model, f'pretrain/{args.model}_clip_visual.pth')
        all_model['clip'] = result_model['state_dict']
        torch.save(all_model, 'pretrain/{}_clip_weights.pth'.format(args.model))