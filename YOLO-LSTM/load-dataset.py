from roboflow import Roboflow
def main():
    rf = Roboflow(api_key="")
    project = rf.workspace("mario-agent-workspace").project("mario-dataset-o65mb")
    version = project.version(1)
    dataset = version.download("yolo26")

main()