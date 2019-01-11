from data.getters import get_sun_tags, get_name_enum_mapping


def beutify_class_name(class_name):
    parent_name = Path(class_name).parent.name
    if 'indoor' in class_name:
        name = f'{parent_name}.i'
    elif 'outdoor' in class_name:
        name = f'{parent_name}.o'
    else:
        name = Path(class_name).name
    return str(name)


if __name__ == '__main__':
    print('text')

    get_sun_tags()
    get_name_enum_mapping()
