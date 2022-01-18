import numpy as np
import random
import sys
import os 
# setting path to parent directory

DEPTH_LIMIT = 5
sys.path.append(os.getcwd())

from BaseAI import BaseAI
from Grid import Grid

OPPONENT = lambda player: 3 - player

class EasyAI(BaseAI):

    def __init__(self, initial_position = None) -> None:
        super().__init__()
        self.pos = initial_position
        self.player_num = None

    def setPosition(self, new_pos: tuple):
        self.pos = new_pos
    
    def getPosition(self):
        return self.pos 

    def setPlayerNum(self, num):
        self.player_num = num

    def getMove(self, grid):
        """ Returns a random, valid move """
        
        # find all available moves 
        current_grid = grid.clone()
        if (len(current_grid.get_neighbors(current_grid.find(self.player_num), only_available=True))==1):
            return current_grid.get_neighbors(current_grid.find(self.player_num), only_available=True)[0]
        position,hvalue = self.moveMinimax(None, current_grid, DEPTH_LIMIT,True)
        return position

    def getTrap(self, grid : Grid):

        """EasyAI throws randomly to the immediate neighbors of the opponent"""
        # edge case: if player wins by moving before trap is thrown, throw randomly
        current_grid = grid.clone()
        if (len(current_grid.get_neighbors(current_grid.find(3-self.player_num), only_available=True))==1):
            return current_grid.get_neighbors(current_grid.find(3-self.player_num), only_available=True)[0]
        position,hvalue = self.trapMinimax(None, current_grid, DEPTH_LIMIT,True)
        return position
    
    def moveHeuristic(self, grid:Grid):
        current_grid = grid.clone()
        self_pos = current_grid.find(self.player_num)
        moves = current_grid.get_neighbors(grid.find(self.player_num),only_available=True)
        sum = 0
        for move in moves:
            current_grid.move(move,self.player_num)
            num_move_self = len(current_grid.get_neighbors(move, only_available=True))
            sum += num_move_self
            current_grid.move(self_pos,self.player_num)
        return sum

    def TrapHeuristic(self, grid:Grid):
        current_grid = grid.clone()
        moves = current_grid.get_neighbors(grid.find(3 - self.player_num),only_available=True)
        oppo_pos = current_grid.find(3 - self.player_num)
        sum = 0
        for move in moves:
            current_grid.move(move,3-self.player_num)
            num_move_oppo = len(current_grid.get_neighbors(oppo_pos, only_available=True))
            sum+=num_move_oppo
            current_grid.move(oppo_pos,3-self.player_num)
        return -sum
    
    def moveMinimax(self, position, grid:Grid, depth,isMaximizing):
        current_grid = grid.clone()
        if depth == 0 or len(current_grid.get_neighbors(current_grid.find(self.player_num), only_available=True)) == 0:
            return position, self.moveHeuristic(current_grid)
        if isMaximizing:
            self_pos = current_grid.find(self.player_num)
            self_possible_moves = current_grid.get_neighbors(self_pos, only_available= True)
            for move in self_possible_moves:
                self_pos = current_grid.find(self.player_num)
                current_grid.move(move,self.player_num)
                position,eval = self.moveMinimax(move,current_grid,depth-1,False)
        else:
            move_position = position 
            self_pos = current_grid.find(self.player_num)
            oppo_possible_moves = current_grid.get_neighbors(self_pos, only_available=True)
            for trap in oppo_possible_moves:
                current_grid.trap(trap)
                position,eval = self.moveMinimax(move_position, current_grid, depth-1,True)


    def trapMinimax(self, position, grid:Grid, depth,isMaximizing):
        current_grid = grid.clone()
        if depth == 0 or len(current_grid.get_neighbors(current_grid.find(3-self.player_num), only_available=True)) == 0:
            return position,self.TrapHeuristic(current_grid)
        if isMaximizing:
            oppp_pos = current_grid.find(3 - self.player_num)
            self_possible_trap = current_grid.get_neighbors(oppp_pos, only_available= True)
            for intended_trap in self_possible_trap:
                current_grid.trap(intended_trap)
                position,eval = self.trapMinimax(intended_trap, current_grid, depth-1,True)
        else:
            trap_position = position
            oppp_pos = current_grid.find(3 - self.player_num)
            oppo_possible_moves = current_grid.get_neighbors(oppp_pos, only_available=True)
            for move in oppo_possible_moves:
                current_grid.move(move,3 - self.player_num)
                position,eval = self.TrapHeuristic(trap_position,current_grid,depth-1,False)
    