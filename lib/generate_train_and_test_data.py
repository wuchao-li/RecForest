
from __future__ import print_function

import os

import time

import random
import multiprocessing as mp

import numpy as np

#inp is the file name of orignal data file
#train the the file path to write in used to trian
# test the the file path to write in used to test
# number is the user number to testest, number)
def cut(_input,_train,_test,_validation,_item_vec,_number):
    user_behav = dict()
    user_ids = list()
    item_ids = set()
    with open(_input,'r') as f:
        for line in f:
            arr = line.split(',')
            if len(arr) != 5:
                break
            if int(arr[1]) not in item_ids:
                item_ids.add(int(arr[1]))
            if int(arr[0]) not in user_behav:
                user_ids.append(int(arr[0]))
                user_behav[int(arr[0])] = list()

            user_behav[int(arr[0])].append(line)

    random.shuffle(user_ids)
    test_user_ids = user_ids[:_number]
    validation_user_ids=user_ids[_number:_number*2]
    train_user_ids = user_ids[_number*2:]

    #write train data set
    with open(_train, 'w') as f:
        for uid in train_user_ids:
            for line in user_behav[uid]:
                f.write(line)

    with open(_test, 'w') as f:
        for uid in test_user_ids:
            for line in user_behav[uid]:
                f.write(line)
    with open(_validation, 'w') as f:
        for uid in validation_user_ids:
            for line in user_behav[uid]:
                f.write(line)
    # re label usr id and item id which make the initial id is 0
    user_ids={id:i for i,id in enumerate(user_behav)}
    item_ids={id:i for i,id in enumerate(item_ids)}
    print('user number {}, item number {}'.format(len(user_ids),len(item_ids)))

    # train_user_ids={id:i for i,id in enumerate(train_user_ids)}
    # train_item_his = dict()
    # with open(_input,'r') as f:
    #     for line in f:
    #         arr = line.split(',')
    #         if len(arr) != 5:
    #             break   
    #         item = item_ids[int(arr[1])]
    #         if item not in train_item_his:
    #             train_item_his[item] = list()
    #         if int(arr[0]) not in test_user_ids and int(arr[0]) not in validation_user_ids:
    #             #print(1) 
    #             train_item_his[item].append(train_user_ids[int(arr[0])])
    # train_item_vec = np.zeros([len(item_ids), len(train_user_ids)])
    # for i in range(len(item_ids)):
    #     item_his = train_item_his[i]
    #     train_item_vec[i][item_his] = 1.0
    
    # np.save(_item_vec,train_item_vec)

    return user_ids, item_ids

def _read(raw_file,test_record_num):
    train_data_file,test_data_file,validation_data_file='train.dat','test.dat','validation.dat'
    path_seg=raw_file.split('/')
    prefix=''
    if len(path_seg)>1  :
        for seg in path_seg[:-1]:
                prefix+=seg+'/'
    user_ids,item_ids=cut(raw_file,\
                          prefix+'raw_'+train_data_file,\
                          prefix+'raw_'+test_data_file,
                          prefix + 'raw_' + validation_data_file,\
                          prefix + 'train_item_vec',\
                          test_record_num)

    behavior_dict = dict()  # record the behavior type, 5 types, key is the type, value is the id of the type
    train_sample = dict()  # record user id, item id, cate_id, behavior_id, timestampe. key the liks 'USERID', value is a array.
    test_sample = dict()
    validation_sample=dict()
    user_id = list()
    item_id = list()
    cat_id = list()
    behav_id = list()
    timestamp = list()

    start = time.time()
    itobj = zip([prefix+'raw_'+train_data_file, prefix+'raw_'+test_data_file, prefix+'raw_'+validation_data_file],\
                [train_sample, test_sample,validation_sample])
    for filename, sample in itobj:
        with open(filename, 'r') as f:
            for line in f:
                arr = line.split(',')
                if len(arr) != 5:
                    break
                user_id.append(user_ids[int(arr[0])])
                item_id.append(item_ids[int(arr[1])])
                cat_id.append(int(float(arr[2])))
                if arr[3] not in behavior_dict:
                    i = len(behavior_dict)
                    behavior_dict[arr[3]] = i
                behav_id.append(behavior_dict[arr[3]])
                timestamp.append(int(arr[4]))
            sample["USERID"] = np.array(user_id)
            sample["ITEMID"] = np.array(item_id)
            sample["CATID"] = np.array(cat_id)
            sample["BEHAV"] = np.array(behav_id)
            sample["TS"] = np.array(timestamp)

            user_id = []
            item_id = []
            cat_id = []
            behav_id = []
            timestamp = []

    #write train data set
    '''
    with open(prefix+'processed_'+train_data_file, 'w') as f:
        for user_id,item_id,cat_id,behav_id,timestamp in zip(train_sample["USERID"],train_sample["ITEMID"],
                                                             train_sample["CATID"],train_sample["BEHAV"],train_sample["TS"]):

            f.write(str(user_id)+','+str(item_id)+','+str(cat_id)+','+str(behav_id)+','+str(timestamp)+'\n')

    with open(prefix+'processed_'+test_data_file, 'w') as f:
        for user_id,item_id,cat_id,behav_id,timestamp in zip(test_sample["USERID"],test_sample["ITEMID"],
                                                             test_sample["CATID"],test_sample["BEHAV"],test_sample["TS"]):

            f.write(str(user_id)+','+str(item_id)+','+str(cat_id)+','+str(behav_id)+','+str(timestamp)+'\n')

    with open(prefix+'processed_'+validation_data_file, 'w') as f:
        for user_id,item_id,cat_id,behav_id,timestamp in zip(validation_sample["USERID"],validation_sample["ITEMID"],
                                                             validation_sample["CATID"],validation_sample["BEHAV"],validation_sample["TS"]):

            f.write(str(user_id)+','+str(item_id)+','+str(cat_id)+','+str(behav_id)+','+str(timestamp)+'\n')
    '''
    print("Read data done, {} train records, {} test records"", elapsed: {}".format(len(train_sample["USERID"]),
                                                                                    len(test_sample["USERID"]),
                                                                                    time.time() - start))
    os.remove(prefix + 'raw_' + train_data_file)
    os.remove(prefix + 'raw_' + test_data_file)
    os.remove(prefix + 'raw_' + validation_data_file)
    # train_sample record user id, item id, cate_id, behavior_id, timestampe. key the liks 'USERID', value is a array.
    # test_sample is like train_sample
    # behavior_dict record the behavior type, 5 types, key is the type, value is the id of the type
    return behavior_dict, train_sample, test_sample,validation_sample,len(user_ids),len(item_ids)

def _gen_user_his_behave(train_sample):

    user_his_behav = dict()
    iterobj = zip(train_sample["USERID"],
                  train_sample["ITEMID"], train_sample["TS"])
    for user_id, item_id, ts in iterobj:
        if user_id not in user_his_behav:
            user_his_behav[user_id] = list()
        user_his_behav[user_id].append((item_id, ts))

    for _, value in user_his_behav.items():
        value.sort(key=lambda x: x[1])
    # user_his_behav is a dict, key is userid and value is a list [(itemId,timestamp)], list is ascendent by timestamp
    return user_his_behav

def _gen_discriminator_samples(train_sample,discriminator_instances_file,train_sample_seg_cnt=400,\
                               parall=5,seq_len=70,min_seq_len=6):
    user_his_behav = _gen_user_his_behave(train_sample)#
    print("user_his_behav len: {}".format(len(user_his_behav)))
    users = list(user_his_behav.keys())
    process = []
    pipes = []
    job_size = int(len(user_his_behav) / parall)
    if len(user_his_behav) % parall != 0:
        parall += 1
    for i in range(parall):
        #a, b = mp.Pipe()
        #pipes.append(a)
        p = mp.Process(
            target=_partial_gen_discriminator_samples,
            args=(users[i * job_size: (i + 1) * job_size],
                  user_his_behav,
                  '{}.part_{}'.format(discriminator_instances_file, i),seq_len,min_seq_len)
        )
        process.append(p)
        p.start()
    '''
    t=0
    for pipe in pipes:
        (count) = pipe.recv()
        t+=count
    print('{} discriminator training instances!'.format(t))
    '''
    for p in process:
        p.join()
    #print('total instances is {}'.format(count))
    # Merge partial files
    with open(discriminator_instances_file, 'w') as f:
        for i in range(parall):
            filename = '{}.part_{}'.format(discriminator_instances_file, i)
            with open(filename, 'r') as f1:
                f.write(f1.read())

            os.remove(filename)

    # Split train sample to segments
    _split_train_sample(discriminator_instances_file,train_sample_seg_cnt)



def _partial_gen_discriminator_samples(users,user_his_behav, filename,seq_len,min_seq_len):
    # user_his_behav is a dict, key is userid and value is a list [(itemId,timestamp)], list is ascendent by timestamp
    # filename is the file to be written into, i.e. sample file
    count = 0
    with open(filename, 'w') as f:
        for user in users:
            value = user_his_behav[user]  # the clicked item id for user
            count = len(value)  # the item number of the user to click
            if count < min_seq_len:
                continue
            arr = [-1 for i in range(seq_len - min_seq_len)] + \
                  [v[0] for v in value]
            # arr is [0,0...,0,itemid0,itemid1,....]
            # each line user_id|item1,item2,...,item(self.lem-1)|label
            for i in range(len(arr) - seq_len+1):
                sample = arr[i: i + seq_len-1]
                f.write('{}|'.format(user))  # sample id
                f.write("{}|".format(",".join([str(v) for v in sample])))
                #f.write("{}".format(sample[-1]))  # label, no ts
                f.write("{}".format(",".join([str(v) for v in arr[i+seq_len-1:i+seq_len-1+int(seq_len/5)]])))
                f.write('\n')
                count += 1
    #pipe.send((count))

def _partial_gen_train_sample(users,
                              user_his_behav, filename,seq_len,min_seq_len, pipe):
    #user_his_behav is a dict, key is userid and value is a list [(itemId,timestamp)], list is ascendent by timestamp
    #filename is the file to be written into, i.e. sample file
    stat = dict()# record the frequency of the each item 's appearance
    count=0
    with open(filename, 'w') as f:
        for user in users:
            value = user_his_behav[user]# the clicked item id for user
            count = len(value)# the item number of the user to click
            if count < min_seq_len:
                continue
            arr = [-1 for i in range(seq_len - min_seq_len)] + \
                  [v[0] for v in value]
            #arr is [0,0...,0,itemid0,itemid1,....]
            #each line user_id|item1,item2,...,item(self.lem-1)|label
            for i in range(len(arr) - seq_len + 1):
                sample = arr[i: i + seq_len]
                f.write('{}|'.format(user))  # sample id
                f.write("{}|".format(",".join([str(v) for v in sample[:-1]])))
                f.write("{}".format(sample[-1]))  # label, no ts
                f.write('\n')
                count+=1
                if sample[-1] not in stat:
                    stat[sample[-1]] = 0
                stat[sample[-1]] += 1
    pipe.send((stat,count))

def _gen_train_sample(train_sample,train_instances_file,test_sample=None,validation_sample=None,\
                      train_sample_seg_cnt=400,parall=5,seq_len=70,min_seq_len=6):
    #user_his_behav is a dict, key is userid and value is a list [(itemId,timestamp)], list is ascendent by timestamp
    user_his_behav = _gen_user_his_behave(train_sample)#
    print("user_his_behav len: {}".format(len(user_his_behav)))

    users = list(user_his_behav.keys())
    process = []
    pipes = []
    job_size = int(len(user_his_behav) / parall)
    if len(user_his_behav) % parall != 0:
        parall += 1
    for i in range(parall):
        a, b = mp.Pipe()
        pipes.append(a)
        p = mp.Process(
            target=_partial_gen_train_sample,
            args=(users[i * job_size: (i + 1) * job_size],
                  user_his_behav,
                  '{}.part_{}'.format(train_instances_file, i),seq_len,min_seq_len,b)
        )
        process.append(p)
        p.start()

    stat = dict()# record the frequency of the each item 's appearance
    t=0
    for pipe in pipes:
        (st,count) = pipe.recv()
        t+=count
        for k, v in st.items():
            if k not in stat:
                stat[k] = 0
            stat[k] += v

    for p in process:
        p.join()
    #print('total instances is {}'.format(count))
    # Merge partial files
    with open(train_instances_file, 'w') as f:
        for i in range(parall):
            filename = '{}.part_{}'.format(train_instances_file, i)
            with open(filename, 'r') as f1:
                f.write(f1.read())

            os.remove(filename)

        if test_sample is not None:
            user_his_behav = _gen_user_his_behave(test_sample)
            for user, value in user_his_behav.items():
                if len(value)/2 + 1 < min_seq_len:
                    continue

                #mid = int(len(value) / 2)
                mid=int(len(value)/2 + 1)

                left = value[:mid]#[-seq_len + 1:]

                arr = [-1 for i in range(seq_len - min_seq_len)] + \
                      [v[0] for v in left]
                # arr is [0,0...,0,itemid0,itemid1,....]
                # each line user_id|item1,item2,...,item(self.lem-1)|label
                for i in range(len(arr) - seq_len + 1):
                    sample = arr[i: i + seq_len]
                    f.write('{}|'.format(user))  # sample id
                    f.write("{}|".format(",".join([str(v) for v in sample[:-1]])))
                    f.write("{}".format(sample[-1]))  # label, no ts
                    f.write('\n')
        if validation_sample is not None:
            user_his_behav = _gen_user_his_behave(validation_sample)
            for user, value in user_his_behav.items():
                if len(value) / 2 + 1 < min_seq_len:
                    continue

                # mid = int(len(value) / 2)
                mid = int(len(value) / 2 + 1)

                left = value[:mid]  # [-seq_len + 1:]

                arr = [-1 for i in range(seq_len - min_seq_len)] + \
                      [v[0] for v in left]
                # arr is [0,0...,0,itemid0,itemid1,....]
                # each line user_id|item1,item2,...,item(self.lem-1)|label
                for i in range(len(arr) - seq_len + 1):
                    sample = arr[i: i + seq_len]
                    f.write('{}|'.format(user))  # sample id
                    f.write("{}|".format(",".join([str(v) for v in sample[:-1]])))
                    f.write("{}".format(sample[-1]))  # label, no ts
                    f.write('\n')

    # Split train sample to segments
    _split_train_sample(train_instances_file,train_sample_seg_cnt)
    return stat
def _split_train_sample(train_instances_file,train_sample_seg_cnt=400):
    segment_filenames = []
    segment_files = []
    for i in range(train_sample_seg_cnt):
        filename = "{}_{}".format(train_instances_file, i)
        segment_filenames.append(filename)
        segment_files.append(open(filename, 'w'))

    with open(train_instances_file, 'r') as fi:
        for line in fi:
            i = random.randint(0, train_sample_seg_cnt - 1)# train_sample_seg_cnt is 400
            segment_files[i].write(line)

    for f in segment_files:
        f.close()

    os.remove(train_instances_file)

    # Shuffle
    num=0
    for fn in segment_filenames:
        lines = []
        with open(fn, 'r') as f:
            for line in f:
                lines.append(line)
        random.shuffle(lines)
        num+=len(lines)
        with open(fn, 'w') as f:
            for line in lines:
                f.write(line)
    print('number of training instance is {}'.format(num))

def _gen_test_sample(test_sample,test_instances_file,seq_len=70,min_seq_len=6):
    # user_his_behav is a dict, key is userid and value is a list [(itemId,timestamp)], list is ascendent by timestamp
    user_his_behav =_gen_user_his_behave(test_sample)
    with open(test_instances_file, 'w') as f:
        for user, value in user_his_behav.items():
            if len(value) / 2 + 1 < min_seq_len:
                continue

            #mid = int(len(value) / 2)
            mid = int(len(value) / 2 + 1)

            left = value[:mid][-seq_len + 1:]

            #left also need to be put into train instances

            right = value[mid:]
            left = [-1 for i in range(seq_len - len(left) - 1)] + \
                   [l[0] for l in left]
            #arr is [0,0...,0,itemid0,itemid1,....]
            #sample id | group id | features || label | (# label is 1.0)
            #其中features为;分隔的Key @ Value序列
            f.write('{}|'.format(user))  # sample id
            f.write("{}|".format(",".join([str(v) for v in left])))
            labels = ','.join(['{}'.format(item[0]) for item in right])
            f.write('{}'.format(labels))  # test_unit_id is ‘test_unit_id’
            f.write('\n')


if __name__=="__main__":
    '''
    behavior_dict, train_sample, test_sample=_read('data/mock/mock.dat', 'train.dat', 'test.dat', 20)#20 is the test users
    # write the training instance into different train_sample_seg_cnt files， avoid that a file is too large
    #stat record the click frequency of each item
    #seq_len=20 min that 19 behaviors and one label
    stat=_gen_train_sample(train_sample,'./data/train_instances',train_sample_seg_cnt=10,parall=3,seq_len=20,min_seq_len=3)
    _gen_test_sample(test_sample,'./data/test_instances',seq_len=20,min_seq_len=3)
    _init_tree(train_sample, test_sample, stat, kv_file=None)
    '''
