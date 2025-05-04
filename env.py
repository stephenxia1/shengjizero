import numpy as np
import random

CARDMAP = ["3♤", "4♤", "5♤", "6♤", "7♤", "8♤", "9♤", "T♤", "J♤", "Q♤", "K♤", "A♤",
           "3♥", "4♥", "5♥", "6♥", "7♥", "8♥", "9♥", "T♥", "J♥", "Q♥", "K♥", "A♥",
          "3◆", "4◆", "5◆", "6◆", "7◆", "8◆", "9◆", "T◆", "J◆", "Q◆", "K◆", "A◆",
          "3♧", "4♧", "5♧", "6♧", "7♧", "8♧", "9♧", "T♧", "J♧", "Q♧", "K♧", "A♧",
          "2♤", "2♥", "2◆", "2♧", "LJ", "HJ"]

class Env:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.hands = np.zeros((4, 54), dtype=int)
        self.played_cards = np.zeros((4, 54), dtype=int)
        self.landlord = np.random.randint(0, 4)
        self.points = 0
        self.lead_player = self.landlord
        self.current_player = self.landlord
        self.lead_suit = 0
        self.kitty = np.zeros(54, dtype=int)
        self.current_trick = np.zeros((4, 54), dtype=int)
        self.done = False

        self.deck = np.random.permutation(54)
        for i in range(4):
            for j in range(12):
                self.hands[i][self.deck[13*i + j]] += 1
        for i in range(6):
            self.kitty[self.deck[-i-1]] += 1 
        
        self.observation_space = self.getObs(self.current_player)
        self.action_space = np.zeros(54)

    def getValidActions(self, playerNum):
        if self.lead_player == playerNum:
            return self.hands[playerNum]
        elif self.lead_suit == 3:
            if self.hands[playerNum][36:].sum() == 0:
                return self.hands[playerNum]
            else:
                valid = self.hands[playerNum].copy()
                valid[:36] = 0
                return valid
        else:
            if self.hands[playerNum][12*self.lead_suit:12*self.lead_suit+12].sum() == 0:
                return self.hands[playerNum]
            else:
                valid = self.hands[playerNum].copy()
                valid[12*self.lead_suit+12:] = 0
                valid[:12*self.lead_suit] = 0
                return valid

    def getPoints(self, totalcards):
        return 5*(totalcards[2]+totalcards[14]+totalcards[26]+totalcards[38]) + 10*(totalcards[7]+totalcards[19]+totalcards[31]+totalcards[43]+totalcards[10]+totalcards[22]+totalcards[34]+totalcards[46])

    def step(self, action):
        assert self.getValidActions(self.current_player)[action]==1, f"Invalid action {action} for player {self.current_player}"
        if self.lead_player == self.current_player:
            self.lead_suit = min(3, action // 12)
        self.hands[self.current_player][action] -= 1
        self.played_cards[self.current_player][action] += 1
        self.current_trick[self.current_player][action] += 1
        self.current_player = (self.current_player + 1) % 4

        # print(self.current_trick.sum(), " cards played in this trick")
        if self.current_trick.sum() == 4:
            winner = self.lead_player
            for i in range(3):
                if self.compareCards(self.current_trick[(self.lead_player + i + 1) % 4].argmax(), 
                                     self.current_trick[winner].argmax()) == 1:
                    winner = (self.lead_player + i + 1) % 4
            if winner % 2 != self.landlord % 2:
                self.points += self.getPoints(np.sum(self.current_trick, axis=0))
            self.lead_player = winner%4
            self.current_player = self.lead_player
            self.current_trick = np.zeros((4, 54), dtype=int)

            # print("cards left in hands:", np.sum(self.hands))

            if np.sum(self.hands) == 0:
                if winner % 2 != self.landlord % 2:
                    # print("Kitty:", [CARDMAP[j] for j in np.where(self.kitty==1)[0]], "Kitty Points:", 2*self.getPoints(self.kitty))
                    self.points += 2*self.getPoints(self.kitty)
                self.done = True
        
        return self.getObs(self.current_player)

    def get_reward(self, playerNum):
        if self.done:
            ranks = np.ceil(self.points/20).astype(int)
            if self.points == 0:
                ranks = 0

            if playerNum % 2 == self.landlord % 2:
                return -ranks
            else:
                return ranks
        else:
            return 0

    def getObs(self, playerNum):
        return np.concatenate([self.hands[playerNum%4].flatten(), 
                              self.played_cards[playerNum%4].flatten(),
                              self.played_cards[(playerNum+1)%4].flatten(), 
                              self.played_cards[(playerNum+2)%4].flatten(), 
                              self.played_cards[(playerNum+3)%4].flatten(), 
                              self.current_trick[playerNum%4].flatten(),
                              self.current_trick[(playerNum+1)%4].flatten(), 
                              self.current_trick[(playerNum+2)%4].flatten(), 
                              self.current_trick[(playerNum+3)%4].flatten(), 
                              np.array(self.points).flatten(),
                              np.array(self.lead_player).flatten(),
                              np.array(self.current_player).flatten(),
                              np.array(self.lead_suit).flatten(),
                              np.array(self.landlord).flatten()
                              ])

    def compareCards(self, card1, card2):
        '''
        1 if card1 > card2, 0 if card1 == card2, -1 if card1 < card2
        '''
        if card1 == card2:
            return 0
        if card1 >= 36 and card2 >= 36:
            if card1 in [48, 49, 50] and card2 in [48, 49, 50]:
                return 0
            else:
                return 1 if card1 > card2 else -1
        if card1 >= 36 and card2 < 36:
            return 1
        if card1 < 36 and card2 >= 36:
            return -1
        if card1 // 12 == self.lead_suit and card2 // 12 == self.lead_suit:
            return 1 if card1 > card2 else -1
        if card1 // 12 == self.lead_suit and card2 // 12 != self.lead_suit:
            return 1
        if card1 // 12 != self.lead_suit and card2 // 12 == self.lead_suit:
            return -1
        

# test = Env()
# turn = 0
# print("Landlord:", test.landlord)
# while (not test.done):
#     print("------ROUND", turn, "START------", "Lead Player:", test.lead_player, "Current Player:", test.current_player)
#     # print("Lead Player:", test.lead_player)
#     for i in range(4):
#         # print("Current Player:", test.current_player)
#         # print(test.getValidActions(test.current_player))
#         # print("valid actions:", [CARDMAP[j] for j in np.where(test.getValidActions(test.current_player) == 1)[0]])
#         action = np.random.choice(range(54), p=test.getValidActions(test.current_player)/np.sum(test.getValidActions(test.current_player)))
#         print(test.current_player, "Action:", CARDMAP[action], "   Hand:", [CARDMAP[j] for j in np.where(test.hands[test.current_player] == 1)[0]], " Valid:", [CARDMAP[j] for j in np.where(test.getValidActions(test.current_player) == 1)[0]])
#         test.step(action)
    
#     turn+=1
    
#     print("Winner:", test.lead_player)
#     print("Points:", test.points)