import copy


class XTBParam:
    def __init__(self, content: str = None, patterns: list = None, values: list = None):
        self.__content = None
        self.__patterns = None
        self.__values = None
        self.__parm = None
        self.__state = self.set_parameters(content, patterns, values)

    def set_parameters(self, content: str, patterns: list, values: list) -> bool:
        if (len(patterns) == len(values)):
            self.__content = content
            self.__patterns = patterns
            self.__values = values
            self.__parm = self.__get_parameters()
            return True
        else:
            return False

    def __get_parameters(self) -> str:
        parm = copy.deepcopy(self.__content)
        for i in range(len(self.__patterns)):
            parm = parm.replace(self.__patterns[i], "{:10.6f}".format(self.__values[i]))
        return parm

    def get_parameters(self):
        return self.__parm

    def print_param_file(self, file: str):
        if self.__state:
            pfile = open(file, 'w')
            pfile.write(self.__parm)
            pfile.close()
        return self.__state