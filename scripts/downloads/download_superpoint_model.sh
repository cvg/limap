echo "Downloading Superpoint model..."
mkdir -p limap/point2d/superpoint/weights
wget ftp://b1ueber2y.me/LIMAP/SuperPoint/superpoint_v1.pth
mv superpoint_v1.pth limap/point2d/superpoint/weights
