class dataset:
    def __init__(self, dName=None, dDescription=None):
        self.name = dName
        self.description = dDescription


class method:
    def __init__(self, mName=None, mDescription=None):
        self.name = mName
        self.description = mDescription


class evaluate:
    def __init__(self, eName=None, eDescription=None):
        self.name = eName
        self.description = eDescription


class setting:
    def __init__(self, sName=None, sDescription=None):
        self.name = sName
        self.description = sDescription
        self.dataset = None
        self.method = None
        self.result = None
        self.evaluate = None


class result:
    def __init__(self, rName=None, rDescription=None):
        self.name = rName
        self.description = rDescription
        self.data = None
