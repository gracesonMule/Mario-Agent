from roboflow import Roboflow
def main():
    rf = Roboflow(api_key="")
    project = rf.workspace("ethan-ortega-zqedq").project("mario-dataset")
    version = project.version(6)
    dataset = version.download("yolo26")

main()