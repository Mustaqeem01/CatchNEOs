import pyperclip as p
#print(dir(p))
with open("neo_v2.csv",'r') as f:
	p.copy(f.read())