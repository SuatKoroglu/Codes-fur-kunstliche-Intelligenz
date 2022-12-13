
from constraint import *
problem = Problem()

nationality = ["English", "Spanish", "Ukrainian", "Norwegian", "Japanese"]
pet = ["dog", "snails", "fox", "horse", "zebra"]
cigarette = ["Old Gold", "Kools", 
"Chesterfields", "Lucky Strike", "Parliaments"]
colour = ["red", "green", "yellow", "blue", "ivory"]
beverage = ["coffee", "milk", "orange juice", "water", "tea"]

criteria = nationality + pet + cigarette + colour + beverage
problem.addVariables(criteria,[1,2,3,4,5])

problem.addConstraint(AllDifferentConstraint(), nationality)
problem.addConstraint(AllDifferentConstraint(), pet)
problem.addConstraint(AllDifferentConstraint(), cigarette)
problem.addConstraint(AllDifferentConstraint(), colour)
problem.addConstraint(AllDifferentConstraint(), beverage)

problem.addConstraint(lambda e, r: e == r, ["English","red"])
problem.addConstraint(lambda s, d: s == d, ("Spanish","dog"))
problem.addConstraint(lambda c, g: c == g, ("coffee","green"))
problem.addConstraint(lambda u, t: u == t, ("Ukrainian","tea"))
problem.addConstraint(lambda g, i: g-i == 1, ("green","ivory"))
problem.addConstraint(lambda o, s: o == s, ("Old Gold","snails"))
problem.addConstraint(lambda k, y: k == y, ("Kools","yellow"))
problem.addConstraint(InSetConstraint([3]), ["milk"])
problem.addConstraint(InSetConstraint([1]), ["Norwegian"])
problem.addConstraint(lambda c, f: abs(c-f) == 1, ("Chesterfields","fox"))
problem.addConstraint(lambda k, h: abs(k-h) == 1, ("Kools","horse"))
problem.addConstraint(lambda l, o: l == o, ["Lucky Strike","orange juice"])
problem.addConstraint(lambda j, p: j == p, ["Japanese","Parliaments"])
problem.addConstraint(lambda k, h: abs(k-h) == 1, ("Norwegian","blue"))

solution = problem.getSolutions()[0]


sol = dict(solution)




for i in range(1,6):
    for x in solution:
        if solution[x]== i :
            if x == "water":
                keys = [k for k, v in sol.items() if v == i]
                for f in keys:
                    if f in nationality:
                        print(f ,"drinks" ,x, "in house:", str(i))
                
            if x == "zebra":
                keys = [k for k, v in sol.items() if v == i]
                for f in keys:
                    if f in nationality:
                        print(f, "has",x, "in house:", str(i))

