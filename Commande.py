# Width=1280 #Récupere la valeur de ton programme
# Commande=0 #Initialisation à mettre dans le start

def SetCom(Input,Width):
    W = [0, Width / 10, 2 * Width / 10, 3 * Width / 10, 4 * Width / 10, 45*Width/100, 5 * Width / 10, 55*Width/100, 6 * Width / 10, 7 * Width / 10, 8 * Width / 10, 9 * Width / 10, Width]
    S = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0, 0.2, 0.4, 0.6, 0.8, 1]

    for i in range(0, len(W)-1):
        if (Input >= W[i] and Input <= W[i+1]):
            return(S[i])

def SetSpeed(Input, Width): #A appeler dans ton tick, il faudrait aussi plot la valeur et potentiellement faire un vecteur qui part
    # du centre et va dans la direction de la commande
    Command = 0
    Command=(SetCom(Input, Width)+4*Command)/5
    if (Command>=0):
        if Command > 0:
            #print('Turn Right')
            pass
        else:
            #print('Centralized')
            pass
        Speed_w1=0.15+Command*0.15
        Speed_w2=0.15
    if(Command<0):
        print('Turn Left')
        Speed_w1=0.15
        Speed_w2=0.15-Command*0.15
    return(Speed_w1,Speed_w2,Command)
