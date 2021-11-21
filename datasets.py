import numpy as np
import tensorflow as tf
from tqdm import tqdm
from os.path import join

def create_dataset_from_encoded(read_path, max_length, index = None):
    encoded_path = join(read_path, 'encoded.txt')
    f = open(encoded_path, 'r', encoding = 'utf-8')
    print('Reading encoded data from \"{}\"...'.format(encoded_path))
    lines = f.read().splitlines()

    uttr_emots_path = join(read_path, 'uttr_emots.npy')
    print('Reading emotions from \"{}\"...'.format(uttr_emots_path))
    uttr_emots = np.load(uttr_emots_path)
    uttr_emots = np.argsort(uttr_emots, axis = 1)

    # RoBERTa uses 1 as the padding value
    inputs = np.ones((len(lines), max_length), dtype = np.int32)
    input_segments = np.ones((len(lines), max_length), dtype = np.int32)
    input_emots = np.zeros((len(lines), max_length), dtype = np.int32)

    targets_i = np.ones((len(lines), max_length), dtype = np.int32)
    targets_r = np.ones((len(lines), max_length), dtype = np.int32)
    target_segments = np.ones((len(lines), max_length), dtype = np.int32)
    target_emots = np.zeros(len(lines), dtype = np.int32)

    n = 0
    for i, line in tqdm(enumerate(lines), total = len(lines)):
        inp_str, inp_seg_str, tar_str, tar_seg_str, idx_str = line.split('\t')
        j, k = [int(s) for s in idx_str.split(',')]

        inp_ids = [int(s) for s in inp_str.split(',')]
        inp_seg_ids = [int(s) for s in inp_seg_str.split(',')]
        tar_ids = [int(s) for s in tar_str.split(',')]
        tar_seg_ids = [int(s) for s in tar_seg_str.split(',')]

        seg_id = 0
        for x in range(len(inp_seg_ids)):
            if inp_seg_ids[x] != seg_id:
                j += 1
                seg_id = inp_seg_ids[x]
            input_emots[i,x] = uttr_emots[j,-1]
        target_emots[i] = uttr_emots[k,-1]

        inputs[i,:len(inp_ids)] = inp_ids
        input_segments[i,:len(inp_seg_ids)] = inp_seg_ids
        targets_i[i,:len(tar_ids)-1] = tar_ids[:-1]
        targets_r[i,:len(tar_ids)-1] = tar_ids[1:]
        target_segments[i,:len(tar_seg_ids)-1] = tar_seg_ids[:-1]

    f.close()

    if index is not None:
        inputs = inputs[index]
        input_segments = input_segments[index]
        input_emots = input_emots[index]
        targets_i = targets_i[index]
        targets_r = targets_r[index]
        target_segments = target_segments[index]
        target_emots = target_emots[index]

    return (tf.data.Dataset.from_tensor_slices(inputs),
            tf.data.Dataset.from_tensor_slices(input_segments),
            tf.data.Dataset.from_tensor_slices(input_emots),
            tf.data.Dataset.from_tensor_slices(targets_i),
            tf.data.Dataset.from_tensor_slices(targets_r),
            tf.data.Dataset.from_tensor_slices(target_segments),
            tf.data.Dataset.from_tensor_slices(target_emots)), inputs.shape[0]

def create_os_datasets(tokenizer, path, buffer_size, batch_size, max_length):
    print('Vocabulary size is {}.'.format(tokenizer.vocab_size))

    train_dataset, _ = create_dataset_from_encoded(join(path, 'train'), max_length)
    val_dataset, _ = create_dataset_from_encoded(join(path, 'valid'), max_length)
    test_dataset, _ = create_dataset_from_encoded(join(path, 'test'), max_length)

    train_dataset = tf.data.Dataset.zip(train_dataset).shuffle(buffer_size).batch(batch_size)
    val_dataset = tf.data.Dataset.zip(val_dataset).batch(batch_size)
    test_dataset = tf.data.Dataset.zip(test_dataset).batch(batch_size)

    return train_dataset, val_dataset, test_dataset

def create_os_test_dataset(tokenizer, path, batch_size, max_length, index = None):
    print('Vocabulary size is {}.'.format(tokenizer.vocab_size))
    test_dataset, N = create_dataset_from_encoded(path, max_length, index)
    test_dataset = tf.data.Dataset.zip(test_dataset).batch(batch_size)
    return test_dataset, N

def create_ed_datasets(tokenizer, path, buffer_size, batch_size, max_length, index = None):
    print('Vocabulary size is {}.'.format(tokenizer.vocab_size))

    def create_dataset(read_path, cascade = True):
        print('Reading data from \"{}\"...'.format(read_path))

        if not cascade:
            with open(join(read_path, 'prompts.txt'), 'r', encoding = 'utf-8') as f:
                all_prompts = f.read().splitlines()
            with open(join(read_path, 'context_emots.txt'), 'r', encoding = 'utf-8') as f:
                all_context_emots = f.read().splitlines()
        else:
            all_prompts = []
            all_context_emots = []

        with open(join(read_path, 'uttrs.txt'), 'r', encoding = 'utf-8') as f:
            uttrs = f.read().splitlines()

        # For test set, we randomly choose a turn to be target.
        dialogs_file = 'dialogs_partial.txt' if not cascade else 'dialogs.txt'
        with open(join(read_path, dialogs_file), 'r', encoding = 'utf-8') as f:
            dialogs = [(int(i) for i in line.split(',')) for line in f.read().splitlines()]

        uttr_emots = np.load(join(read_path, 'uttr_emots.npy'))
        assert len(uttrs) == uttr_emots.shape[0]
        uttr_emots = np.argsort(uttr_emots, axis = 1)

        SOS_ID = tokenizer.encode('<s>')[0]
        EOS_ID = tokenizer.encode('</s>')[0]

        # RoBERTa uses 1 as the padding value
        # For RoBERTa style input: <s> u1 </s> </s> u2 </s> </s> u3 </s> ...
        inputs = np.ones((len(uttrs), max_length), dtype = np.int32)

        # These three are always associated with RoBERTa style input
        input_segments = np.zeros((len(uttrs), max_length), dtype = np.int32)
        input_emots = np.zeros((len(uttrs), max_length), dtype = np.int32)
        target_segments = np.zeros((len(uttrs), max_length), dtype = np.int32)
        target_emots = np.zeros(len(uttrs), dtype = np.int32)

        # These two are always the same for any input style: <s> target </s>
        targets_i = np.ones((len(uttrs), max_length), dtype = np.int32)
        targets_r = np.ones((len(uttrs), max_length), dtype = np.int32)

        n = 0
        indices = []
        for ind, (s, t) in tqdm(enumerate(dialogs), total = len(dialogs)):
            if t - s < 2:
                continue

            if cascade:
                uttr_ids = tokenizer.encode(uttrs[s])

                inp_ids = [SOS_ID] + uttr_ids + [EOS_ID]
                inp_seg_ids = [0] * (len(uttr_ids) + 2)
                inp_emots = [uttr_emots[s,-1]] * (len(uttr_ids) + 2)

                for i in range(s + 1, t):
                    u = ' '.join(uttrs[s:i])
                    if len(u.split()) > max_length: break

                    uttr_ids = tokenizer.encode(uttrs[i])
                    tar_ids = [SOS_ID] + uttr_ids + [EOS_ID]
                    tar_seg_ids = [(i - s) % 2] * (len(uttr_ids) + 2)

                    if (len(inp_ids) <= max_length and len(tar_ids) - 1 <= max_length):
                        inputs[n,:len(inp_ids)] = inp_ids
                        input_segments[n,:len(inp_seg_ids)] = inp_seg_ids
                        input_emots[n,:len(inp_ids)] = inp_emots
                        target_emots[n] = uttr_emots[i,-1]
                        targets_i[n,:len(tar_ids)-1] = tar_ids[:-1]
                        targets_r[n,:len(tar_ids)-1] = tar_ids[1:]
                        target_segments[n,:len(tar_seg_ids)] = tar_seg_ids
                        n += 1

                    inp_ids += ([EOS_ID] + uttr_ids + [EOS_ID])
                    inp_seg_ids += [(i - s) % 2] * (len(uttr_ids) + 2)
                    inp_emots += [uttr_emots[i,-1]] * (len(uttr_ids) + 2)
            else:
                u = ' '.join(uttrs[s:t-1])
                if len(u.split()) > max_length: continue

                uttr_ids = tokenizer.encode(uttrs[s])
                inp_ids = [SOS_ID] + uttr_ids + [EOS_ID]
                inp_seg_ids = [0] * (len(uttr_ids) + 2)
                inp_emots = [uttr_emots[s,-1]] * (len(uttr_ids) + 2)
                for i in range(s + 1, t - 1):
                    uttr_ids = tokenizer.encode(uttrs[i])
                    inp_ids += ([EOS_ID] + uttr_ids + [EOS_ID])
                    inp_seg_ids += [(i - s) % 2] * (len(uttr_ids) + 2)
                    inp_emots += [uttr_emots[i,-1]] * (len(uttr_ids) + 2)

                uttr_ids = tokenizer.encode(uttrs[t - 1])
                tar_ids = [SOS_ID] + uttr_ids + [EOS_ID]
                tar_seg_ids = [(t - s - 1) % 2] * (len(uttr_ids) + 2)

                if (len(inp_ids) <= max_length and len(tar_ids) - 1 <= max_length):
                    inputs[n,:len(inp_ids)] = inp_ids
                    input_segments[n,:len(inp_seg_ids)] = inp_seg_ids
                    input_emots[n,:len(inp_ids)] = inp_emots
                    target_emots[n] = uttr_emots[t-1,-1]
                    targets_i[n,:len(tar_ids)-1] = tar_ids[:-1]
                    targets_r[n,:len(tar_ids)-1] = tar_ids[1:]
                    target_segments[n,:len(tar_seg_ids)] = tar_seg_ids
                    indices.append(ind)
                    n += 1

        print('Created dataset with {} examples.'.format(n))
        print('Number of indices: {}'.format(len(indices)))

        prompts = []
        for ind in indices:
            prompts.append((all_context_emots[ind], all_prompts[ind]))

        inputs = inputs[:n,:]
        input_segments = input_segments[:n,:]
        input_emots = input_emots[:n,:]
        targets_i = targets_i[:n,:]
        targets_r = targets_r[:n,:]
        target_segments = target_segments[:n,:]
        target_emots = target_emots[:n]

        if not cascade and index is not None:
            inputs = inputs[index]
            input_segments = input_segments[index]
            input_emots = input_emots[index]
            targets_i = targets_i[index]
            targets_r = targets_r[index]
            target_segments = target_segments[index]
            target_emots = target_emots[index]

        return (tf.data.Dataset.from_tensor_slices(inputs),
                tf.data.Dataset.from_tensor_slices(input_segments),
                tf.data.Dataset.from_tensor_slices(input_emots),
                tf.data.Dataset.from_tensor_slices(targets_i),
                tf.data.Dataset.from_tensor_slices(targets_r),
                tf.data.Dataset.from_tensor_slices(target_segments),
                tf.data.Dataset.from_tensor_slices(target_emots)), prompts, inputs.shape[0]

    train_dataset, _, _ = create_dataset(join(path, 'train'))
    val_dataset, _, _ = create_dataset(join(path, 'valid'))
    test_dataset, prompts, N = create_dataset(join(path, 'test'), cascade = False)

    train_dataset = tf.data.Dataset.zip(train_dataset).shuffle(buffer_size).batch(batch_size)
    val_dataset = tf.data.Dataset.zip(val_dataset).batch(batch_size)
    test_dataset = tf.data.Dataset.zip(test_dataset).batch(batch_size)

    return train_dataset, val_dataset, test_dataset, prompts, N
