# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os
import json
import errno
import tensorflow as tf
import copy
import random
import numpy
import scipy
import math
import threading
import time
import PIL
from random import shuffle
from shutil import copyfile

flags = tf.app.flags
flags.DEFINE_string('dir',
                    "/home/keyong/Downloads/studio_watson",
                    """the source folder which conutains image and annotatoin should have structure 
.
└──samples
└──templates.json  """)

FLAGS = flags.FLAGS

num_rows=900
num_cols=900
def mkdir_p(path):
    try:
        os.makedirs(path, 0o770)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def create_fake_image(sku_id,studio_path,studio_file,photos_dir, anno_dir, fake_jpg):
    print ("create fake image from %s/%s to %s/%s" % (studio_path, studio_file,photos_dir, fake_jpg))
    num_rows=600
    num_cols=600
    bg_r = numpy.random.random((num_rows,num_cols))
    bg_g = numpy.random.random((num_rows,num_cols))
    bg_b = numpy.random.random((num_rows,num_cols))

    img=scipy.ndimage.imread( os.path.join (studio_path, studio_file))
    if img.shape[2] == 4:
        r,g,b,a = numpy.rollaxis(img,-1)
        #aa=numpy.clip((a+127)/255.0,0,1.0)
        aa=a*1.0
        rr=r/255.0
        gg=g/255.0
        bb=b/255.0
    else:
        r,g,b = numpy.rollaxis(img,-1)
        rr=r/255.0
        gg=g/255.0
        bb=b/255.0

    #img=numpy.dstack((rr.astype("uint8"),gg.astype("uint8"),bb.astype("uint8")))

    #w=random.uniform(0.3,0.75)
    h=img.shape[0]
    w=img.shape[1]
    ratio=random.uniform(0.5,0.85)
    #ratio=1.0
    new_ratio=num_rows*ratio/max(h,w)
    #new_h= (int) (math.ceil(new_ratio*h))
    #new_w= (int) (math.ceil(new_ratio*w))
    rr=scipy.ndimage.zoom(rr, new_ratio)
    gg=scipy.ndimage.zoom(gg, new_ratio)
    bb=scipy.ndimage.zoom(bb, new_ratio)
    new_h=rr.shape[0]
    new_w=rr.shape[1]

    #scaled_img=numpy.dstack((rr.astype("uint8"),gg.astype("uint8"),bb.astype("uint8")))

    #scipy.misc.imsave(os.path.join(photos_dir,fake_jpg),scaled_img)
    x_1=(int) (num_cols-new_w)
    x_1=random.randint(0,x_1)
    y_1=(int) (num_rows-new_h)
    y_1=random.randint(0,y_1)
    x_2= num_cols-new_w-x_1
    y_2= num_rows-new_h-y_1
    ###>>> a = [[1, 2], [3, 4]]
    ###>>> np.lib.pad(a, ((3, 2), (2, 3)), 'minimum')
    # array([[1, 1, 1, 2, 1, 1, 1],
    #        [1, 1, 1, 2, 1, 1, 1],
    #        [1, 1, 1, 2, 1, 1, 1],
    #        [1, 1, 1, 2, 1, 1, 1],
    #        [3, 3, 3, 4, 3, 3, 3],
    #        [1, 1, 1, 2, 1, 1, 1],
    #        [1, 1, 1, 2, 1, 1, 1]])

    rr=numpy.lib.pad(rr, ((y_1,y_2),(x_1,x_2)),'constant', constant_values=(0.0))
    gg=numpy.lib.pad(gg, ((y_1,y_2),(x_1,x_2)),'constant', constant_values=(0.0))
    bb=numpy.lib.pad(bb, ((y_1,y_2),(x_1,x_2)),'constant', constant_values=(0.0))
    all=(rr*255).astype("uint32")+(gg*255).astype("uint32")+(bb*255).astype("uint32")
    #padded_img=numpy.dstack((rr.astype("uint8"),gg.astype("uint8"),bb.astype("uint8")))
    cc=numpy.where(all==0 )
    rr[cc]+=bg_r[cc]
    gg[cc]+=bg_g[cc]
    bb[cc]+=bg_b[cc]

    rr*=255
    gg*=255
    bb*=255
    padded_img=numpy.dstack((rr.astype("uint8"),gg.astype("uint8"),bb.astype("uint8")))
    scipy.misc.imsave(os.path.join(photos_dir,fake_jpg),padded_img)
    bndboxes=[]
    #{"x":242.98105263157896,"y":1638.2652631578949,
    # "w":234.43157894736845,"h":698.2616673007506,
    # "id":"235402","strokeStyle":"#3399FF","fillStyle":"#00FF00"}
    box={}
    box["x"]=x_1
    box["y"]=y_1
    box["w"]=new_w
    box["h"]=new_h
    box["id"]=sku_id
    box["strokeStyle"]="#%02x%02x%02x"%(random.randint(0,255),random.randint(0,255),random.randint(0,255))
    box["fillStyle"]="#00FF00"
    bndboxes.append(box)
    with open(os.path.join(anno_dir, fake_jpg + ".json"),'w') as fp:
        header = '{"version":"1.0.0","company":"idontknow","dataset":"photos","filename":"'
        header +=  fake_jpg + '",' + """
  "image_width":600,"image_height":600,
  "bndboxes":
 """
        footer = "}"
        fp.write(header)
        bbs = json.dumps(bndboxes)
        fp.write(bbs)
        fp.write(footer)


def create_fake_image2(sku_id,sample_img, sample_short_name, bg_img,
                       photos_dir, anno_dir, fake_jpg ):
    #print ("create fake image from %s/%s to %s/%s" % (studio_path, studio_file,photos_dir, fake_jpg))
    #num_rows=900
    #num_cols=900
    if bg_img.shape[0] <= num_rows or bg_img.shape[1] <= num_cols:
        return
    bg_x=numpy.random.randint(0,bg_img.shape[1]-num_cols)
    bg_y=numpy.random.randint(0,bg_img.shape[0]-num_rows)
    time1= time.time()
    my_bg=bg_img[bg_y:(bg_y+num_rows),bg_x:(bg_x+num_cols) ]
    bg_r,bg_g,bg_b = numpy.rollaxis(my_bg,-1)
    time2= time.time()

    new_h=sample_img.shape[0]
    new_w=sample_img.shape[1]
    bndboxes=[]
    img=sample_img

    has_alpha=False
    if img.shape[2] == 4:
        r,g,b,a = numpy.rollaxis(img,-1)
        #aa=numpy.clip((a+127)/255.0,0,1.0)
        rr=r
        gg=g
        bb=b
        has_alpha=True
    else:
        r,g,b = numpy.rollaxis(img,-1)
        rr=r
        gg=g
        bb=b

    time3= time.time()


    #rr=scipy.ndimage.zoom(rr, new_ratio)
    #gg=scipy.ndimage.zoom(gg, new_ratio)
    #bb=scipy.ndimage.zoom(bb, new_ratio)
    rects=[]
    def has_overlap(rt2):
        for rt1 in rects:
            if rt1[0] < rt2[2] and rt1[2] > rt2[0] and rt1[1] < rt2[3] and rt1[3] > rt2[1]:
                return True;
        return False;

    def add_sku(new_img):
        x_1=(int) (num_cols*3/4 -new_w)
        x_1=random.randint((int)(num_cols/4),x_1)

        y_1=(int) (num_rows*3/4-new_h)
        y_1=random.randint(num_rows/4 ,y_1)

        x_2= new_w+x_1
        y_2= new_h+y_1

        if has_overlap((x_1,y_1,x_2,y_2)):
            return False;

        rrr,ggg,bbb=rr.copy(),gg.copy(),bb.copy()
        if has_alpha:
            all=(rr).astype("uint32")+(gg).astype("uint32")+(bb).astype("uint32")

            cc=numpy.where(all==0)
            list_cc=copy.deepcopy(list(cc))
            list_cc[0]+=y_1
            list_cc[1]+=x_1

            rrr[cc]=bg_r[list_cc]
            ggg[cc]=bg_g[list_cc]
            bbb[cc]=bg_b[list_cc]

        #new_img=my_bg.copy()
        sku_img=numpy.dstack((rrr,ggg,bbb))
        new_img[y_1:new_h+y_1,x_1:new_w+x_1]=sku_img


        #{"x":242.98105263157896,"y":1638.2652631578949,
        # "w":234.43157894736845,"h":698.2616673007506,
        # "id":"235402","strokeStyle":"#3399FF","fillStyle":"#00FF00"}
        box={}
        box["x"]=x_1
        box["y"]=y_1
        box["w"]=new_w
        box["h"]=new_h
        box["id"]=sku_id
        box["strokeStyle"]="#%02x%02x%02x"%(random.randint(0,255),random.randint(0,255),random.randint(0,255))
        box["fillStyle"]="#00FF00"
        bndboxes.append(box)
        rects.append((x_1,y_1,x_2,y_2))
        return True

    new_img=my_bg.copy()
    for x in range(30):
        add_sku(new_img)
        if len(rects) > 5:
            break

    scipy.misc.imsave(os.path.join(photos_dir,fake_jpg),new_img)
    with open(os.path.join(anno_dir, fake_jpg + ".json"),'w') as fp:
        header = '{"version":"1.0.0","company":"idontknow","dataset":"photos","filename":"'
        header +=  fake_jpg + '",' + """
  "image_width":900,"image_height":900,
  "bndboxes":
 """
        footer = "}"
        fp.write(header)
        bbs = json.dumps(bndboxes)
        fp.write(bbs)
        fp.write(footer)

    # print(" stage 1:%f, 2:%f, 3:%f : 4:%f, 5:%f, 6:%f " % (time2-time1,
    #                                                           time3-time2,
    #                                                           time4-time3,
    #                                                           time5-time4,
    #                                                           time6-time5,
    #                                                           time7-time6
    #                                                           )
    #       )
    return

def main(_):
    src_dir = FLAGS.dir
    generate_files(src_dir)


class Sku:
    def __init__(self, sku_id, full_path):
        self.sku_id=sku_id
        #self.full_path=full_path
        img=scipy.ndimage.imread(full_path)
        self.has_alpha=(img.shape[2] == 4)
        if self.has_alpha:
            r,g,b,a = numpy.rollaxis(img,-1)
            #aa=numpy.clip((a+127)/255.0,0,1.0)
            self.rr=r
            self.gg=g
            self.bb=b
        else:
            self.rr,self.gg,self.bb = numpy.rollaxis(img,-1)

def save_fake_image(new_img, bndboxes, photos_dir, anno_dir, fake_jpg):
    scipy.misc.imsave(os.path.join(photos_dir,fake_jpg),new_img)
    with open(os.path.join(anno_dir, fake_jpg + ".json"),'w') as fp:
        header = '{"version":"1.0.0","company":"idontknow","dataset":"photos","filename":"'
        header +=  fake_jpg + '",' + """
  "image_width":%d, "image_height":%d,
  "bndboxes":
 """%(num_cols,num_rows)
        fp.write(header)
        bbs = json.dumps(bndboxes)
        fp.write(bbs)
        fp.write("}")

def make_sample_list(scaled_sample_dir):
    sample_list=[]
    for sku_id in [f for f in os.listdir(scaled_sample_dir)
                          if os.path.isdir(os.path.join(scaled_sample_dir, f))
                   ]:
        my_path=os.path.join(scaled_sample_dir, sku_id)
        for sku_file in [f for f in os.listdir(my_path) if os.path.isfile(os.path.join(my_path,f))]:
            sku=Sku(sku_id, os.path.join(my_path,sku_file))
            sample_list.append(sku)
    return sample_list

def crop_bg(bg_img):
    if bg_img.shape[0] <= num_rows or bg_img.shape[1] <= num_cols:
        return None
    bg_x=numpy.random.randint(0,bg_img.shape[1]-num_cols)
    bg_y=numpy.random.randint(0,bg_img.shape[0]-num_rows)
    my_bg=bg_img[bg_y:(bg_y+num_rows),bg_x:(bg_x+num_cols) ].copy()
    return my_bg

def add_inner_sku(my_bg,bndboxes,rects, sku):
    new_h=sku.rr.shape[0]
    new_w=sku.rr.shape[1]

    x_1=(int) (num_cols*2/3 -new_w)
    x_start=(int)(num_cols/3)
    x_1=random.randint(x_start,max(x_1,x_start+1))

    y_1=(int) (num_rows*2/3-new_h)
    y_start=(int)(num_rows/3)
    y_1=random.randint(y_start,max(y_1,y_start+1))

    x_2= new_w+x_1
    y_2= new_h+y_1
    def has_overlap():
        for rt in rects:
            if rt[0] < x_2 and rt[2] > x_1 and rt[1] < y_2 and rt[3] > y_1:
                return True;
        return False;

    # def has_overlap(rt2):
    #     for rt1 in rects:
    #         if rt1[0] < rt2[2] and rt1[2] > rt2[0] and rt1[1] < rt2[3] and rt1[3] > rt2[1]:
    #             return True;
    #     return False;
    #
    # if has_overlap((x_1,y_1,x_2,y_2)):
    if has_overlap():
        return False;

    new_img=my_bg
    bg_r,bg_g,bg_b = numpy.rollaxis(my_bg,-1)
    rrr,ggg,bbb=sku.rr.copy(),sku.gg.copy(),sku.bb.copy()
    if sku.has_alpha:
        all=(sku.rr).astype("uint32")+(sku.gg).astype("uint32")+(sku.bb).astype("uint32")
        cc=numpy.where(all==0)
        list_cc=copy.deepcopy(list(cc))
        list_cc[0]+=y_1
        list_cc[1]+=x_1

        rrr[cc]=bg_r[list_cc]
        ggg[cc]=bg_g[list_cc]
        bbb[cc]=bg_b[list_cc]

    sku_img=numpy.dstack((rrr,ggg,bbb))
    new_img[y_1:new_h+y_1,x_1:new_w+x_1]=sku_img


    #{"x":242.98105263157896,"y":1638.2652631578949,
    # "w":234.43157894736845,"h":698.2616673007506,
    # "id":"235402","strokeStyle":"#3399FF","fillStyle":"#00FF00"}
    box={}
    box["x"]=x_1
    box["y"]=y_1
    box["w"]=new_w
    box["h"]=new_h
    box["id"]=sku.sku_id
    box["strokeStyle"]="#%02x%02x%02x"%(random.randint(0,255),random.randint(0,255),random.randint(0,255))
    box["fillStyle"]="#00FF00"
    bndboxes.append(box)
    rects.append((x_1,y_1,x_2,y_2))
    return True

def add_outer_sku(my_bg,bndboxes,rects, sku):
    new_h=sku.rr.shape[0]
    new_w=sku.rr.shape[1]
    x_1,y_1=0,0
    if new_h >=300 or new_w >= 300:
        return True

    def has_overlap():
        for rt in rects:
            if rt[0] < x_1+new_w and rt[2] > x_1 and rt[1] < y_1 + new_h and rt[3] > y_1:
                return True;
        return False;
    good_coord=False
    for x in range(4):
        bar_indicator= random.randint(0,4)
        if bar_indicator == 0:
            #left bar
            x_1=(int)(num_cols/3) - new_w
            x_1=random.randint(0,x_1)
            y_1=random.randint(0,num_rows-new_h)
        elif bar_indicator == 1:
            #try top bar
            x_1=random.randint((int)(num_cols/3) ,(int)(num_cols*2/3) - new_w)
            y_1=random.randint(0,(int)(num_rows/3)-new_h)
        elif bar_indicator == 2:
            #try bottom bar
            x_1=random.randint((int)(num_cols/3) ,(int)(num_cols*2/3) -new_w)
            y_1=random.randint((int)(num_rows*2/3),num_rows-new_h)
        elif bar_indicator == 3:
            #try right bar
            x_1=random.randint((int)(num_cols*2/3), num_cols-new_w)
            y_1=random.randint(0,num_rows-new_h)

        if not has_overlap():
            good_coord=True
            break

    if not good_coord:
        return False

    x_2= new_w+x_1
    y_2= new_h+y_1

    new_img=my_bg
    bg_r,bg_g,bg_b = numpy.rollaxis(my_bg,-1)
    rrr,ggg,bbb=sku.rr.copy(),sku.gg.copy(),sku.bb.copy()
    if sku.has_alpha:
        all=(sku.rr).astype("uint32")+(sku.gg).astype("uint32")+(sku.bb).astype("uint32")
        cc=numpy.where(all==0)
        list_cc=copy.deepcopy(list(cc))
        list_cc[0]+=y_1
        list_cc[1]+=x_1

        rrr[cc]=bg_r[list_cc]
        ggg[cc]=bg_g[list_cc]
        bbb[cc]=bg_b[list_cc]

    sku_img=numpy.dstack((rrr,ggg,bbb))
    #print ("y_1=%d,y_2=%d,x_1=%d,x_2=%d"%(y_1,y_2,x_1,x_2))
    new_img[y_1:y_2,x_1:x_2]=sku_img


    #{"x":242.98105263157896,"y":1638.2652631578949,
    # "w":234.43157894736845,"h":698.2616673007506,
    # "id":"235402","strokeStyle":"#3399FF","fillStyle":"#00FF00"}
    box={}
    box["x"]=x_1
    box["y"]=y_1
    box["w"]=new_w
    box["h"]=new_h
    box["id"]=sku.sku_id
    box["strokeStyle"]="#%02x%02x%02x"%(random.randint(0,255),random.randint(0,255),random.randint(0,255))
    box["fillStyle"]="#00FF00"
    bndboxes.append(box)
    rects.append((x_1,y_1,x_2,y_2))
    return True

def scale_sku_by_ratio(ratios, path, sku_file,sub_sample_dir):

    full_path=os.path.join (path, sku_file)
    with tf.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        with PIL.Image.open(encoded_jpg_io) as image:
            w, h = image.size

    suffix=".jpg"
    if sku_file[-3:].upper()=="PNG":
        suffix=".png"

    for idx, ratio in enumerate(ratios):
        new_ratio = math.sqrt(num_cols* num_rows * ratio*ratio/(h*w))
        key       = os.path.join(sub_sample_dir , sku_file+ "_" + str(idx)+suffix)
        new_width = new_ratio * w
        new_height= new_ratio * h
        cmd="convert '" + full_path + "' -resize " + str(new_width) + "x" + str(new_height) + " '" + key +"'"
        os.system(cmd)
        #break

def generate_empty_annotation(src_dir, src_file, dst_dir,dst_file,anno_dir):
    #scipy.misc.imsave(os.path.join(photos_dir,fake_jpg),new_img)
    copyfile(os.path.join(src_dir,src_file),os.path.join(dst_dir,dst_file))
    bndboxes=[]
    with open(os.path.join(anno_dir, src_file + ".json"),'w') as fp:
        header = '{"version":"1.0.0","company":"idontknow","dataset":"photos","filename":"'
        header +=  dst_file+ '",' + """
  "image_width":%d, "image_height":%d,
  "bndboxes":
 """%(num_cols,num_rows)
        fp.write(header)
        bbs = json.dumps(bndboxes)
        fp.write(bbs)
        fp.write("}")

def generate_files(src_dir):
    mkdir_p(os.path.join(src_dir, "photos", "Annotations"))
    mkdir_p(os.path.join(src_dir, "scaled_samples"))
    scaled_sample_dir=os.path.join(src_dir, "scaled_samples")
    annotations_dir = os.path.join(src_dir, "photos", "Annotations")
    bg_dir = os.path.join(src_dir, "background")
    pure_bg_dir = os.path.join(src_dir, "pure_background")
    photos_dir = os.path.join(src_dir, "photos")
    sample_dir = os.path.join(src_dir, "samples")
    samples_dir = [f for f in os.listdir(sample_dir)
                         if os.path.isdir(os.path.join(sample_dir, f))
                     ]
    image_count=0
    running_threads=[]
    #ratios=numpy.linspace(0.10,200.0/num_cols,20)
    ratios=numpy.linspace(100.0/num_cols,200.0/num_cols,5)

    start =  time.time()
    # scale the sku files
    for folder in samples_dir:
        path=os.path.join(sample_dir,folder)
        sub_sample_dir=os.path.join(scaled_sample_dir,folder)
        if os.path.exists(sub_sample_dir):
            continue

        def scale_sku_func(l_sub_sample_dir,l_path):
            mkdir_p(l_sub_sample_dir)
            for sku_file in [f for f in os.listdir(l_path) if os.path.isfile(os.path.join(l_path,f))]:
                scale_sku_by_ratio(ratios, l_path, sku_file, l_sub_sample_dir)

        t=threading.Thread(name="thread for "+folder,target=scale_sku_func,
                           args=(""+sub_sample_dir,""+path))
        running_threads.append(t)
        t.start()


    for t in running_threads:
        t.join()
    end =time.time()
    print ("scaling sample files takes %f sec"%(end-start))
    #return

    sample_list = make_sample_list(scaled_sample_dir)
    shuffle(sample_list)
    batch_size=len(sample_list)*2

    def thread_func(bg_file,bg_img,base):
        cropped_bg = crop_bg(bg_img)
        if cropped_bg is None:
           return
        boxes=[] #for json
        rects=[]
        idx=0
        print("thread %s: starts from base:%d" %(threading.current_thread().name, base))

        outer_idx=0

        for sku in sample_list:
            added=False
            for x in range(40):
                if add_inner_sku(cropped_bg,boxes,rects,sku):
                    added=True
                    break

            if not added and len(rects) > 0:
                sku2=sample_list[outer_idx]
                rects2=[]
                added=False
                for x in range(20):
                    if add_outer_sku(cropped_bg,boxes,rects2,sku2):
                        added=True
                        outer_idx+=1
                        if outer_idx >= len(sample_list):
                            #print(">>>>>>>one round of filling outer bar is done!!!<<<<<<")
                            outer_idx=0
                        sku2=sample_list[outer_idx]

                save_fake_image(cropped_bg,boxes, photos_dir, annotations_dir,
                                bg_file+"_%06d.jpg"%(idx+base))
                if (idx+1) % 100 ==0:
                    print("%s: created %d images" % (threading.current_thread().name, idx+1))
                idx += 1
                cropped_bg = crop_bg(bg_img)
                boxes=[] #for json
                rects=[]
                #break
                add_inner_sku(cropped_bg,boxes,rects,sku)

        if len(rects) > 0:
            save_fake_image(cropped_bg,boxes, photos_dir, annotations_dir,
                            bg_file+"_%06d.jpg"%(idx+base))
            #if (idx+1) % 100 ==0:
            #    print("%s: created %d images" % (threading.current_thread().name, idx+1))
            idx+=1

    start_count=0
    print("generating background image from %s" % (pure_bg_dir))
    #bg_imgs=[]
    for bg_file in [f for f in os.listdir(pure_bg_dir) if os.path.isfile(os.path.join(pure_bg_dir,f))]:
        #bg_img=scipy.ndimage.imread(os.path.join (bg_dir, bg_file))
        #if bg_img.shape[0] <= num_rows and bg_img.shape[1] <= num_cols:
        generate_empty_annotation(pure_bg_dir,bg_file,photos_dir,bg_file,annotations_dir)
    print("generating fake image based on %s" % (bg_dir))
    #bg_imgs=[]
    for bg_file in [f for f in os.listdir(bg_dir) if os.path.isfile(os.path.join(bg_dir,f))]:
        bg_img=scipy.ndimage.imread(os.path.join (bg_dir, bg_file))
        if bg_img.shape[0] <= num_rows and bg_img.shape[1] <= num_cols:
            continue
            #bg_imgs.append((bg_file,bg_img))

        #for (bg_file, bg_img) in bg_imgs:
        another_bg_img=bg_img
        t=threading.Thread(name="%-40s"%(bg_file),target=thread_func,
                           args=(""+bg_file,another_bg_img,start_count))
        running_threads.append(t)
        t.start()
        start_count+=batch_size

        if len(running_threads) >=10:
            for t in running_threads:
                t.join()
            print("10 background images has been taken!" )
            running_threads=[]
        #break

    for t in running_threads:
        t.join()
    end = time.time()
    print ("take %f sec"%(end-start))


if __name__ == '__main__':
    tf.app.run()
    #generate_files("/home/keyong/Downloads/studio_data")

