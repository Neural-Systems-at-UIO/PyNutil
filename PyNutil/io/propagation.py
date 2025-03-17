import math, re


def propagate(arr):
    arr = arr.copy()
    for slice in arr:
        if "nr" not in slice:
            slice["nr"] = int(re.search(r"_s(\d+)", slice["filename"]).group(1))

    arr.sort(key=lambda slice: slice["nr"])

    linregs = [LinReg() for i in range(11)]
    count = 0
    for slice in arr:
        if "anchoring" in slice:
            a = slice["anchoring"]
            for i in range(3):
                a[i] += (a[i + 3] + a[i + 6]) / 2
            a.extend(
                [normalize(a, 3) / slice["width"], normalize(a, 6) / slice["height"]]
            )
            for i in range(len(linregs)):
                linregs[i].add(slice["nr"], a[i])
            count += 1

    if count >= 2:
        l = len(arr)
        if not "anchoring" in arr[0]:
            nr = arr[0]["nr"]
            a = [linreg.get(nr) for linreg in linregs]
            orthonormalize(a)
            arr[0]["anchoring"] = a
            count += 1
        if not "anchoring" in arr[l - 1]:
            nr = arr[l - 1]["nr"]
            a = [linreg.get(nr) for linreg in linregs]
            orthonormalize(a)
            arr[l - 1]["anchoring"] = a
            count += 1

        start = 1
        while count < l:
            while "anchoring" in arr[start]:
                start += 1
            next = start + 1
            while not "anchoring" in arr[next]:
                next += 1
            pnr = arr[start - 1]["nr"]
            nnr = arr[next]["nr"]
            panch = arr[start - 1]["anchoring"]
            nanch = arr[next]["anchoring"]
            linints = [LinInt(pnr, panch[i], nnr, nanch[i]) for i in range(len(panch))]
            for i in range(start, next):
                nr = arr[i]["nr"]
                arr[i]["anchoring"] = [linint.get(nr) for linint in linints]
                count += 1
            start = next + 1

        for slice in arr:
            a = slice["anchoring"]
            orthonormalize(a)
            v = a.pop()
            u = a.pop()
            for i in range(3):
                a[i + 3] *= u * slice["width"]
                a[i + 6] *= v * slice["height"]
                a[i] -= (a[i + 3] + a[i + 6]) / 2
    return arr


class LinInt:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def get(self, x):
        return self.y1 + (self.y2 - self.y1) * (x - self.x1) / (self.x2 - self.x1)


class LinReg:
    def __init__(self):
        self.n = 0
        self.Sx = 0
        self.Sy = 0
        self.Sxx = 0
        self.Sxy = 0

    def add(self, x, y):
        self.n += 1
        self.Sx += x
        self.Sy += y
        self.Sxx += x * x
        self.Sxy += x * y
        if self.n >= 2:
            self.b = (self.n * self.Sxy - self.Sx * self.Sy) / (
                self.n * self.Sxx - self.Sx * self.Sx
            )
            self.a = self.Sy / self.n - self.b * self.Sx / self.n

    def get(self, x):
        return self.a + self.b * x


def normalize(arr, idx):
    l = 0
    for i in range(3):
        l += arr[idx + i] * arr[idx + i]
    l = math.sqrt(l)
    for i in range(3):
        arr[idx + i] /= l
    return l


def orthonormalize(arr):
    normalize(arr, 3)
    dot = 0
    for i in range(3):
        dot += arr[i + 3] * arr[i + 6]
    for i in range(3):
        arr[i + 6] -= arr[i + 3] * dot
    normalize(arr, 6)
