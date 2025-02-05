import numpy as np
import skimage
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Board:
    def __init__(self,
                ledLength = 0.65,
                ledWidth = 0.35,
                colNs = [3,6,8,8,8,6,3],
                y1s = [1.04, 1.94, 2.84, 2.84, 2.84, 1.94, 1.04],
                yLens = [2.7, 4.5, 6.3, 6.3, 6.3, 4.5, 2.7],
                r1 = -7.85,
                rL = 6.6,
                LEDIds = [31,25,19,38,32,26,20,14,1,37,39,33,27,21,8,2,13,36,40,34,28,15,9,3,7,30,41,35,22,16,10,4,6,42,29,23,17,11,5,24,18,12],
                ):
        self.ledLength = ledLength
        self.ledWidth = ledWidth
        self.colNs = colNs
        self.y1s = y1s
        self.yLens = yLens
        self.r1 = r1
        self.rL = rL
        self.LEDIds = LEDIds

    def getLedPos(self):
        ys = np.hstack([np.linspace(y1,y1-yL,colN) for y1,yL,colN in zip(self.y1s, self.yLens, self.colNs)])
        xs = np.hstack([[x]*n for x,n in zip(np.linspace(self.r1, self.rL+self.r1, len(self.colNs)), self.colNs)])
        #recenter with origin at the bottom left
        xs = (xs - np.min(xs)) + self.ledLength/2
        ys = (ys - np.min(ys)) + self.ledWidth/2
        return np.vstack([xs,ys]).T
    
    def renderBoardState(self, state, gridRes = 0.05):
        xs, ys = self.getLedPos().T
        gridSize = [np.max(ys)+self.ledWidth/2,np.max(xs)+self.ledLength/2] #mm
        grid = np.zeros([int(gridSize[0]/gridRes),int(gridSize[1]/gridRes)])
        for x,y,v in zip(xs,ys,state):
            rr, cc = skimage.draw.rectangle([(y-self.ledWidth/2)/gridRes,(x-self.ledLength/2)/gridRes], extent = (self.ledWidth/gridRes, self.ledLength/gridRes), shape = grid.shape)
            grid[rr,cc] = v
        return grid
    
    def renderBoardAnimation(self, states, gridRes = 0.05):
        return [self.renderBoardState(state, gridRes) for state in states]
    

    def animateBoard(self, states, framerate, cmap='grey', titles = None):
        rendered = np.array(self.renderBoardAnimation(states))
        fig, ax = plt.subplots(figsize = (2,2))
        im = plt.imshow(rendered[0,:,:], origin = 'lower', cmap = cmap);
        if titles is not None:
            title = ax.set_title(titles[0])
        plt.axis('off');
        def update(frame):
            im.set_data(rendered[frame,:,:])
            if titles is not None:
                title.set_text(titles[frame])
            return im,
        return animation.FuncAnimation(fig, update, frames=len(states), interval=1000/framerate, blit=True)
    
    def ledStates(self, states):
        return states[:,[self.LEDIds.index(i) for i in range(1, len(self.LEDIds) + 1)]]
    
    def decompose(self, states, maxBit = 31):
        ledStates = self.ledStates(states)
        bit_planes = [(ledStates >> bit) & 1 for bit in range(0, np.log2(maxBit+1).astype('int'))]
        return np.array(bit_planes)
    

def gammaCorrect(x, gamma, maxBit = 31):
    return (maxBit*(x/maxBit)**(gamma)).astype('int')

def gammaPercept(x, gamma, maxBit = 31):
    return ((x/maxBit)**(1/gamma)*maxBit).astype('int')

def convertSimToStates(sim, myBoard, maxBit, correctGamma = True, **gammaArgs):
    if correctGamma:
        states = gammaCorrect(np.array([np.round(np.sum(np.round(sim*maxBit/np.max(sim)
        ).astype('int')*np.expand_dims(myBoard.renderBoardState(np.eye(np.sum(myBoard.colNs), np.sum(myBoard.colNs), 0, dtype=int)[n]), axis=0),axis=(1,2))/np.expand_dims(myBoard.renderBoardState(np.eye(np.sum(myBoard.colNs), np.sum(myBoard.colNs), 0, dtype=int)[n]), axis=0).sum()).astype('int') for n in range(np.sum(myBoard.colNs))]).T, **gammaArgs)
    else:
        states = np.array([np.round(np.sum(np.round(sim*maxBit/np.max(sim)
        ).astype('int')*np.expand_dims(myBoard.renderBoardState(np.eye(np.sum(myBoard.colNs), np.sum(myBoard.colNs), 0, dtype=int)[n]), axis=0),axis=(1,2))/np.expand_dims(myBoard.renderBoardState(np.eye(np.sum(myBoard.colNs), np.sum(myBoard.colNs), 0, dtype=int)[n]), axis=0).sum()).astype('int') for n in range(np.sum(myBoard.colNs))]).T
    return states

def enforceState(states, maxBit, correctGamma = True, low=-1, high=1,**gammaArgs):
    states = np.clip(states, low, high)
    states = (states - low)/(high - low)*maxBit
    if correctGamma:
        return gammaCorrect(states, **gammaArgs).astype('int')
    else:
        return states.astype('int')
    
