echo "Downloading SuperGlue models..."
mkdir -p limap/point2d/superglue/weights
wget https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/weights/superglue_indoor.pth?raw=true -O superglue_indoor.pth
wget https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/weights/superglue_outdoor.pth?raw=true -O superglue_outdoor.pth
mv superglue_indoor.pth limap/point2d/superglue/weights/
mv superglue_outdoor.pth limap/point2d/superglue/weights/
