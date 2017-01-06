
./experiments/scripts/faster_rcnn_end2end.sh 0 VGG_CNN_M_1024 pascal_voc

#./tools/train_net.py --gpu 0 --solver models/pascal_voc/VGG_CNN_M_1024/faster_rcnn_end2end/solver.prototxt --weights data/imagenet_models/VGG_CNN_M_1024.v2.caffemodel --imdb voc_2007_trainval --iters 100 --cfg experiments/cfgs/faster_rcnn_end2end.yml

#./tools/train_net.py --gpu 0 --solver models/pascal_voc/VGG_CNN_M_1024/faster_rcnn_end2end/solver.prototxt --imdb voc_2007_trainval --iters 100 --cfg experiments/cfgs/faster_rcnn_end2end.yml

./tools/train_net.py --gpu 0 --solver models/cr/solver.prototxt --imdb voc_2007_trainval --iters 100 --cfg experiments/cfgs/faster_rcnn_end2end.yml
#Wrote snapshot to: /home/ruqiang/github/py-faster-rcnn_ruqiang826/output/faster_rcnn_end2end/voc_2007_trainval/vgg_cnn_m_1024_faster_rcnn_iter_700.caffemodel


#./tools/test_net.py --gpu 0 --def models/pascal_voc/VGG_CNN_M_1024/faster_rcnn_end2end/test.prototxt --net /home/ruqiang/github/py-faster-rcnn_ruqiang826/output/faster_rcnn_end2end/voc_2007_trainval/vgg_cnn_m_1024_faster_rcnn_iter_700.caffemodel --imdb voc_2007_test --cfg experiments/cfgs/faster_rcnn_end2end.yml

./tools/test_net.py --gpu 0 --def models/cr/test.prototxt --net /home/ruqiang/github/py-faster-rcnn_ruqiang826/output/faster_rcnn_end2end/voc_2007_trainval/cr_ai_iter_100.caffemodel --imdb voc_2007_test --cfg experiments/cfgs/faster_rcnn_end2end.yml

#Saving cached annotations to /home/ruqiang/github/py-faster-rcnn_ruqiang826/data/VOCdevkit2007/annotations_cache/annots.pkl


# for dataset 
# 1. put the images in data/mytestdata/VOC2007/JPEGImages.
# 2. put the xml label in data/mytestdata/VOC2007/Annotations
# 3. run the following command to generate trainval.txt and test.txt
ls -1 data/mytestdata/VOC2007/JPEGImages | sed 's/.png//g' | sed 's/.jpg//g' > data/mytestdata/VOC2007/ImageSets/Main/trainval.txt
