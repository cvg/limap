# image_list.txt
First line: ``n_images``

Then follows: ``img_id``, ``image_name`` for each line

# neighbors.txt
First line: ``n_images``

Then follows: ``img_id``, ``neighbors`` for each line

# detections.txt
First line: ``n_images``

For each image, we have a starting line ``img_id``, ``n_segs``, and follow with ``n_segs`` lines of 2d detections ``[x1, y1, x2, y2]``

# alltracks.txt
First line: ``n_tracks``

For each track, there are 5 lines of information. 

The starting line writes ``track_id``, ``n_supporting_segs``, ``n_supporting_images``. 

The second and third lines are the two endpoints in 3D. 

The 4th and 5th line has the corresponding image id and line id for each supporting 2D segment, thus both being ``n_supporting_segs`` long.


