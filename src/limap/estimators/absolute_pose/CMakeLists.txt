set(FOLDER_NAME "absolute_pose")

LIMAP_ADD_SOURCES(
    bindings.cc
    pl_absolute_pose_ransac.h
    pl_absolute_pose_hybrid_ransac.h
    joint_pose_estimator.h joint_pose_estimator.cc
    hybrid_pose_estimator.h hybrid_pose_estimator.cc
)

