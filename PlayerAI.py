import numpy as np
import random
import time
import sys
import os 
from BaseAI import BaseAI
from Grid import Grid


# TO BE IMPLEMENTED
# 
DEPTH_LIMIT = 5

class PlayerAI(BaseAI):

    def __init__(self) -> None:
        # You may choose to add attributes to your player - up to you!
        # wins 14 / 20 vs EasyAI
        super().__init__()
        self.pos = None
        self.player_num = None
    
    def getPosition(self):
        return self.pos

    def setPosition(self, new_position):
        self.pos = new_position 

    def getPlayerNum(self):
        return self.player_num

    def setPlayerNum(self, num):
        self.player_num = num

    def getMove(self, grid: Grid) -> tuple:
        """ 
        YOUR CODE GOES HERE

        The function should return a tuple of (x,y) coordinates to which the player moves.

        It should be the result of the ExpectiMinimax algorithm, maximizing over the Opponent's *Trap* actions, 
        taking into account the probabilities of them landing in the positions you believe they'd throw to.

        Note that you are not required to account for the probabilities of it landing in a different cell.

        You may adjust the input variables as you wish (though it is not necessary). Output has to be (x,y) coordinates.
        
        """
        current_grid = grid.clone()
        if (len(current_grid.get_neighbors(current_grid.find(self.player_num), only_available=True))==1):
            return current_grid.get_neighbors(current_grid.find(self.player_num), only_available=True)[0]
        position,hvalue = self.moveMinimax(None, current_grid, DEPTH_LIMIT,-sys.maxsize, sys.maxsize, True)
        return position
        

    def getTrap(self, grid : Grid) -> tuple:
        """ 
        YOUR CODE GOES HERE

        The function should return a tuple of (x,y) coordinates to which the player *WANTS* to throw the trap.
        
        It should be the result of the ExpectiMinimax algorithm, maximizing over the Opponent's *Move* actions, 
        taking into account the probabilities of it landing in the positions you want. 
        
        Note that you are not required to account for the probabilities of it landing in a different cell.

        You may adjust the input variables as you wish (though it is not necessary). Output has to be (x,y) coordinates.
        
        """
        current_grid = grid.clone()
        if (len(current_grid.get_neighbors(current_grid.find(3-self.player_num), only_available=True))==1):
            return current_grid.get_neighbors(current_grid.find(3-self.player_num), only_available=True)[0]
        position,hvalue = self.trapMinimax(None, current_grid, DEPTH_LIMIT,-sys.maxsize, sys.maxsize, True)
        return position

  
    def moveHeuristic(self, grid:Grid):
        #summing up all the avaliable positions self could move to after making a move
        current_grid = grid.clone()
        self_pos = current_grid.find(self.player_num)
        moves = current_grid.get_neighbors(grid.find(self.player_num),only_available=True)
        sum = len(moves)
        for move in moves:
            current_grid.move(move,self.player_num)
            num_move_self = len(current_grid.get_neighbors(move, only_available=True))
            sum += num_move_self
            current_grid.move(self_pos,self.player_num)
        return sum

    def TrapHeuristic(self, grid:Grid):
        #summing up all the avaliable positions opponent could move to after self throwing a trap
        current_grid = grid.clone()
        moves = current_grid.get_neighbors(grid.find(3 - self.player_num),only_available=True)
        oppo_pos = current_grid.find(3 - self.player_num)
        sum = 0
        for move in moves:
            current_grid.trap(move)
            num_move_oppo = len(current_grid.get_neighbors(oppo_pos, only_available=True))
            sum+=num_move_oppo
            current_grid.setCellValue(move,0)
        return -sum


    def moveOrder(self,moves,grid:Grid):
        sortList = []
        for move in moves:
            sortList.append(len(grid.get_neighbors(move, only_available=True)))
        return [x for _,x in sorted(zip(sortList,moves))]

    def trapOrder(self,player,moves):
        sortList = []
        for move in moves:
            if (0 in move or 6 in move):
                sortList.append(1)
            else:
                sortList.append(0)
        return [x for _,x in sorted(zip(sortList,moves))]

    def moveMinimax(self, position, grid:Grid, depth, alpha, beta,isMaximizing):
        #try to maximizing the Heuristic for moving
        current_grid = grid.clone()
        if depth == 0 or len(current_grid.get_neighbors(current_grid.find(self.player_num), only_available=True)) == 0 or len(current_grid.get_neighbors(current_grid.find(3-self.player_num), only_available=True)) == 0:
            return position, self.moveHeuristic(current_grid)
        if isMaximizing:
            #mocking self move
            max_eval = -sys.maxsize
            self_pos = current_grid.find(self.player_num)
            self_possible_moves = current_grid.get_neighbors(self_pos, only_available= True)
            self_possible_moves = self.moveOrder(self_possible_moves,current_grid)
            for move in self_possible_moves:
                self_pos = current_grid.find(self.player_num)
                current_grid.move(move,self.player_num)
                position,eval = self.moveMinimax(move,current_grid,depth-1,alpha,beta,False)
                #backtracking
                current_grid.move(self_pos,self.player_num)
                max_eval = max(max_eval,eval)
                alpha = max(alpha,eval)
                if beta <= alpha:
                    break
            return position, max_eval
        else:
            #assuming that the opponent can throw trap to wherever he/she wants
            move_position = position 
            min_eval = sys.maxsize
            self_pos = current_grid.find(self.player_num)
            oppo_possible_moves = current_grid.get_neighbors(self_pos, only_available=True)
            oppo_possible_moves = self.trapOrder(self_pos,oppo_possible_moves)
            for trap in oppo_possible_moves:
                current_grid.trap(trap)
                position,eval = self.moveMinimax(move_position, current_grid, depth-1,alpha,beta,True)
                #backtracking
                current_grid.setCellValue(trap,0)
                min_eval = min(min_eval,eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return current_grid.find(self.player_num),min_eval


    def trapMinimax(self, position, grid:Grid, depth, alpha, beta,isMaximizing):
        current_grid = grid.clone()
        if depth == 0 or len(current_grid.get_neighbors(current_grid.find(self.player_num), only_available=True)) == 0 or len(current_grid.get_neighbors(current_grid.find(3-self.player_num), only_available=True)) == 0:
            return position,self.TrapHeuristic(current_grid)
        if isMaximizing:
             #assuming that the self can throw trap to wherever he/she wants
            trap_position = position
            max_eval = -sys.maxsize
            oppp_pos = current_grid.find(3 - self.player_num)
            self_possible_trap = current_grid.get_neighbors(oppp_pos, only_available= True)
            self_possible_trap = self.trapOrder(oppp_pos,self_possible_trap)
            for intended_trap in self_possible_trap:
                current_grid.trap(intended_trap)
                position,eval = self.trapMinimax(intended_trap, current_grid, depth-1,alpha,beta,True)
                #backtracking
                current_grid.setCellValue(intended_trap,0)
                max_eval = max(max_eval,eval)
                alpha = max(alpha,eval)
                if beta <= alpha:
                    break
            return position,max_eval
        else:
            #mocking opponent move
            trap_position = position
            min_eval = sys.maxsize
            oppp_pos = current_grid.find(3 - self.player_num)
            oppo_possible_moves = current_grid.get_neighbors(oppp_pos, only_available=True)
            oppo_possible_moves = self.moveOrder(oppo_possible_moves,current_grid)
            for move in oppo_possible_moves:
                current_grid.move(move,3 - self.player_num)
                position,eval = self.TrapHeuristic(trap_position,current_grid,depth-1,alpha,beta,False)
                #backtracking
                current_grid.move(oppp_pos,3 - self.player_num)
                min_eval = min(min_eval,eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return trap_position,min_eval


def manhattan_distance(position, target):
        return np.abs(target[0] - position[0]) + np.abs(target[1] - position[1])

def throw(player, grid : Grid, intended_position : tuple) -> tuple:
        # compute probability of success, p
        p = 1 - 0.05*(manhattan_distance(grid.find(player), intended_position) - 1)
        return p