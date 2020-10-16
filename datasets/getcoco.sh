#declare -a cocourl=("http://images.cocodataset.org/zips/train2017.zip"
#                "http://images.cocodataset.org/zips/val2017.zip"
#                "http://images.cocodataset.org/zips/test2017.zip"
#                "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
#                "http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip"
#                "http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip"
#                "http://images.cocodataset.org/annotations/image_info_test2017.zip"
#                "http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip"
#                )

declare -a cocourl=("http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
                "http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip"
                "http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip"
                "http://images.cocodataset.org/annotations/image_info_test2017.zip"
                "http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip"
                )

#declare -a outfile=("train2017.zip"
#                "val2017.zip"
#                "test2017.zip"
#                "annotations_trainval2017.zip"
#                "stuff_annotations_trainval2017.zip"
#                "panoptic_annotations_trainval2017.zip"
#                "image_info_test2017.zip"
#                "image_info_unlabeled2017.zip"
#                )

declare -a outfile=("annotations_trainval2017.zip"
                "stuff_annotations_trainval2017.zip"
                "panoptic_annotations_trainval2017.zip"
                "image_info_test2017.zip"
                "image_info_unlabeled2017.zip"
                )

echo "Output path: " "$1"
mkdir $1
cd $1
for i in "${!cocourl[@]}"; do
    echo wget --progress=dot ${cocourl[i]}
    wget ${cocourl[i]}
    echo unzip $1/${outfile[i]}
    unzip $1/${outfile[i]}
done
