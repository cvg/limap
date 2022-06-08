echo "Downloading S2DNet model..."
mkdir -p limap/features/models/checkpoints
wget ftp://b1ueber2y.me/LIMAP/S2DNet/s2dnet_weights.pth
mv s2dnet_weights.pth limap/features/models/checkpoints/

