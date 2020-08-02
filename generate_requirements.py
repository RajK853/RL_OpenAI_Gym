import os


def generate():
    os.system("pigar -P src/ --without-referenced-comments")
    additional_libraries = ["pigar", "notebook"]
    with open("requirements.txt", "a") as req_file:
        req_file.write("\n# Custom libraries\n")
        for library in additional_libraries:
            req_file.write(f"{library}\n")


if __name__ == "__main__":
    generate()
