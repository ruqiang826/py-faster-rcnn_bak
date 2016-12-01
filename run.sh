
./experiments/scripts/faster_rcnn_end2end.sh 0 VGG_CNN_M_1024 pascal_voc

./tools/train_net.py --gpu 0 --solver models/pascal_voc/VGG_CNN_M_1024/faster_rcnn_end2end/solver.prototxt --weights data/imagenet_models/VGG_CNN_M_1024.v2.caffemodel --imdb voc_2007_trainval --iters 700 --cfg experiments/cfgs/faster_rcnn_end2end.yml

#Wrote snapshot to: /home/ruqiang/github/py-faster-rcnn/output/faster_rcnn_end2end/voc_2007_trainval/vgg_cnn_m_1024_faster_rcnn_iter_700.caffemodel

./tools/test_net.py --gpu 0 --def models/pascal_voc/VGG_CNN_M_1024/faster_rcnn_end2end/test.prototxt --net /home/ruqiang/github/py-faster-rcnn/output/faster_rcnn_end2end/voc_2007_trainval/vgg_cnn_m_1024_faster_rcnn_iter_700.caffemodel --imdb voc_2007_test --cfg experiments/cfgs/faster_rcnn_end2end.yml
