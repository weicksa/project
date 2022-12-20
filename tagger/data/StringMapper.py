class StringMapper(object):

    # zu Aufgabe 3
    # angefangen bei 1, für jedes neue feature um eins erhöhen,
    # bei wiederholung von features gleichen index behalten
    def __init__(self):
        self.map = {}
        self.inverse_map = {}
        self.counter = 0

    def lookup(self, s: str) -> int:
        if s in self.map:
            return self.map[s]
        else:
            self.counter += 1
            self.map[s] = self.counter
            self.inverse_map[str(self.counter)] = s
            return self.counter

    def inverseLookup(self, featureIndex: int) -> str:
        return self.inverse_map[str(featureIndex)]

    # hilfreich um im nachhinein interpretieren zu können, welche features
    # besonders gut sind

    def toFile(self, filename: str):
        with open(filename, "w") as file:
            for key in self.map:
                file.write(f"{key} {self.map[key]}\n")

    def fromFile(self, filename: str):
        mapper = StringMapper()

        reconstructed_map = {}
        reconstructed_inverse_map = {}
        with open(filename) as file:
            lines = file.readlines()
            for line in lines:
                spl = line.split()
                reconstructed_map[spl[0]] = spl[1]
                reconstructed_inverse_map[spl[1]] = spl[0]

        mapper.map = reconstructed_map
        mapper.inverse_map = reconstructed_inverse_map

        return mapper
