# Dataset 
You have to download two datasets. The [RBO dataset of articulated objects and interactions](https://zenodo.org/records/1036660#.XsT9gxaxU5k) and the [RBO affordance](https://zenodo.org/records/21414128) dataset. 

## Original interaction data

The original dataset only provides ROS1 rosbags, which are not compatible with newer ROS2 versions. Therefore, you need to convert the ROS1 .bag files to ROS2 .db3. The folder structure should look like

```text
/rbo_dataset
  interactions/     ← index + published transform CSVs; used by evaluation
  interactions2/    ← locally converted ROS 2 .db3 bags; used by filter input
  objects/          ← annotated meshes
  objects_original/ ← meshes from the rbo_dataset. You need to rename them
```


Once downloaded, you can provide the path to the rbo_dataset in the config file

