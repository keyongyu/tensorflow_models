import scipy
import scipy.ndimage
import scipy.misc
import numpy
import os
import argparse
import json
import sys
import threading
import shutil
import time
from matplotlib import pyplot


# def adjust_color_by_mean(src_path, short_name, dst_path):
#     if not os.path.exists(dst_path):
#         os.makedirs(dst_path, 0o770)
#     # if os.path.exists(os.path.join(dst_path, short_name)):
#     #    return
#
#     image = scipy.ndimage.imread(os.path.join(src_path, short_name))
#     if image.shape[2] == 3:
#         mean = image.mean(axis=(0, 1))
#         '''
#         stddev= numpy.nanstd(image,axis=(0,1))
#         image2 = (image - mean) / stddev + [127, 127, 127]
#         '''
#         # rounded_mean_img = (image-mean).astype(int)
#         mean_img = image - mean
#         stddev = numpy.sum(numpy.abs(mean_img), axis=(0, 1)) / (image.shape[0] * image.shape[1])
#         # min=numpy.min(mean)
#         # image2= image - mean + numpy.array([min,min,min])
#
#         # k=1024/stddev
#         # k=k.astype(int)
#         #image2= rounded_mean_img * k /1024 + [128,128,128]
#         image2 = (image - mean) / stddev + [127, 127, 127]
#
#         image2 = image2.clip(0, 255)
#         scipy.misc.imsave(os.path.join(dst_path, short_name), image2)
#     else:
#         print("file %s: non-rgb color space " % short_name)
#         shutil.copyfile(os.path.join(src_path, short_name), os.path.join(dst_path, short_name))
#
#     # image2=image2/255.0
#     # f, axarr= pyplot.subplots(1,2)
#     # axarr[0].imshow(image )
#     # axarr[1].imshow(image2 )
#     # pyplot.show()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--src_image', type=str,
                        default='/home/keyong/Downloads/anno/public/working/PMI_SET2_del_Blurred/photos',
                        help='image folder to adjust color')
    parser.add_argument('--dst_image', type=str,
                        default='/home/keyong/Downloads/anno/public/working/PMI_SET2_del_Blurred/class_photos',
                        help='image folder to save')
    return parser.parse_args()




def extract_skus(src_dir, short_name, bndboxes, dst_dir,classes,lock):
    image = scipy.ndimage.imread(os.path.join(src_dir, short_name))
    if image is None or image.shape[2] != 3:
        return
    for box in bndboxes:
        klass = box["id"]
        folder = os.path.join(dst_dir,klass)
        lock.acquire()
        if classes.has_key(klass):
            idx =classes[klass]

        else:
            idx = classes[klass] = 0
            if not os.path.exists(folder):
                os.makedirs(folder, 0o770)
        idx +=1
        classes[klass]=idx
        lock.release()
        x=int(box["x"])
        y=int(box["y"])
        w=int(box["w"])
        h=int(box["h"])



        crop=image[y:y+h,x:x+w]
        # mean= numpy.mean(crop,axis=(0,1)).astype("uint8")
        # zoom_lvl=230.0/max(crop.shape[0],crop.shape[1])
        # crop2=scipy.ndimage.zoom(crop,[zoom_lvl,zoom_lvl,1.0])
        sku_file_name=os.path.join(folder,"%s_%d.jpg"% (short_name,idx))
        #sku_file_name=os.path.join(folder,"%d.jpg"% (idx))
        #sku=numpy.zeros((256,256,3),"uint8")
        #sku+=mean
        #(h,w,_)=crop2.shape
        #y = int((256-h)/2)
        #x = int((256-w)/2)
        #sku[y:y+h, x:x+w]=crop2
        scipy.misc.imsave(sku_file_name,crop)


def main(args):
    if not os.path.exists(args.dst_image):
       os.makedirs(args.dst_image, 0o770)

    anno_folder=os.path.join(args.src_image,"Annotations")
    files = []
    classes={}
    for example in os.listdir(anno_folder):
        if not os.path.isfile(os.path.join(anno_folder, example)):
            continue
        files.append(example)
    percentage =0

    def worker(short_name,bndboxes,classes,lock):
        extract_skus(args.src_image, short_name, bndboxes, args.dst_image,classes,lock)

    running_threads=[]
    lock = threading.Lock()
    for idx, example in enumerate(files):
        short_name = example[0:-5]
        path = os.path.join(anno_folder, example)
        img_path = os.path.join(anno_folder,  short_name)
        with open(path, 'r') as fid:
            try:
                json_data = json.load(fid)
            except:
                print("fail to load json file:%s" %img_path)
        bndboxes=json_data.get("bndboxes")

        if len(bndboxes) <=0 :
            continue

        t = threading.Thread(name="thread for "+short_name, target=worker,
                           args=(""+short_name, bndboxes,classes, lock))
        running_threads.append(t)
        t.start()
        #extract_skus(args.src_image, short_name, bndboxes, args.dst_image)
        current_perc = idx * 100 // len(files)
        if current_perc != percentage:
            print("%d%% images has been processed" % (current_perc))
            sys.stdout.flush()
            percentage = current_perc

        #if idx >= 0:
        #    break
        if len(running_threads) >= 8:

            for t_idx, t in enumerate(running_threads):
                t.join()

            print("8 background images has been taken!" )
            running_threads=[]


    for t in running_threads:
        t.join()

    running_threads=[]

def make_train_txt(args):
    classes=[]
    for klass in os.listdir(args.dst_image):
        if not os.path.isdir(os.path.join(args.dst_image, klass)):
            continue
        classes.append(klass)
    classes.sort()
    with open(os.path.join(args.dst_image,"train.txt"),"w") as f:
        for idx, klass in enumerate(classes):
            folder = args.dst_image+"/"+ klass
            for sku_file in os.listdir(folder):
                if not os.path.isfile(os.path.join(folder, sku_file)):
                   continue
                f.write("%s/%s %d\n" %(klass, sku_file, idx))




if __name__ == "__main__":
    args = parse_args()
    #main(args)
    make_train_txt(args)
