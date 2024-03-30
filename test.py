from dm_control import mjcf
from pathlib import Path


def main():
    models_dir = Path.cwd() / "_models"
    parent_path = models_dir / "parent.xml"  # model file name
    child_path = models_dir / "child.xml"  # model file name

    parent = mjcf.from_path(str(parent_path))
    print(parent.to_xml_string())

    child = mjcf.from_path(str(child_path))
    print(child.to_xml_string())

    attachement_site = parent.find('site', 'attachment_site')
    print(f"{attachement_site=}")
    attachement_site.attach(child)
    print(parent.to_xml_string())  # I thought that this should print exactly the same as the final generated XML


if __name__ == "__main__":
    main()
