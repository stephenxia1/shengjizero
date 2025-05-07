import numpy as np
import random
from gym.spaces import Box, Discrete

CARDMAP = [
    "3♤","4♤","5♤","6♤","7♤","8♤","9♤","T♤","J♤","Q♤","K♤","A♤",
    "3♥","4♥","5♥","6♥","7♥","8♥","9♥","T♥","J♥","Q♥","K♥","A♥",
    "3◆","4◆","5◆","6◆","7◆","8◆","9◆","T◆","J◆","Q◆","K◆","A◆",
    "3♧","4♧","5♧","6♧","7♧","8♧","9♧","T♧","J♧","Q♧","K♧","A♧",
    "2♤","2♥","2◆","2♧","LJ","HJ"
]

class EnvL:
    def __init__(self):
        # Define observation and action spaces
        self.observation_space = Box(
            low=-float("inf"), high=float("inf"),
            shape=(491,), dtype=np.float32
        )
        self.action_space = Discrete(54)
        # Initialize state
        _obs, _info = self.reset()

    def reset(self):
        # Initialize all game state
        self.hands         = np.zeros((4, 54), dtype=int)
        self.played_cards  = np.zeros((4, 54), dtype=int)
        self.landlord      = np.random.randint(0, 4)
        self.points        = 0
        self.lead_player   = self.landlord
        self.current_player= self.landlord
        self.lead_suit     = 0
        self.kitty         = np.zeros(54, dtype=int)
        self.current_trick = np.zeros((4, 54), dtype=int)
        self.done          = False

        # Deal cards
        self.deck = np.random.permutation(54)
        for i in range(4):
            for j in range(12):
                self.hands[i][ self.deck[13*i + j] ] += 1
        for i in range(6):
            self.kitty[ self.deck[-i-1] ] += 1

        obs = self.getObs(self.current_player)
        info = {'landlord': self.landlord, 'landlord_team': self.landlord % 2}
        return obs, info

    def getValidActions(self, playerNum):
        if self.lead_player == playerNum:
            return self.hands[playerNum]
        elif self.lead_suit == 3:
            if self.hands[playerNum][36:].sum() == 0:
                return self.hands[playerNum]
            valid = self.hands[playerNum].copy()
            valid[:36] = 0
            return valid
        else:
            suit_start = 12 * self.lead_suit
            if self.hands[playerNum][suit_start:suit_start+12].sum() == 0:
                return self.hands[playerNum]
            valid = self.hands[playerNum].copy()
            valid[suit_start+12:] = 0
            valid[:suit_start]   = 0
            return valid

    def step(self, action):
        player = self.current_player
        # Validate and apply action
        assert self.getValidActions(player)[action] == 1, \
               f"Invalid action {action} for player {player}"
        if self.lead_player == player:
            self.lead_suit = min(3, action // 12)
        self.hands[player][action]         -= 1
        self.played_cards[player][action]  += 1
        self.current_trick[player][action] += 1

        # Advance turn
        self.current_player = (player + 1) % 4

        # Resolve trick if 4 cards played
        if self.current_trick.sum() == 4:
            winner = self.lead_player
            for i in range(1, 4):
                idx = (self.lead_player + i) % 4
                if self.compareCards(
                    self.current_trick[idx].argmax(),
                    self.current_trick[winner].argmax()
                ) == 1:
                    winner = idx
            if winner % 2 != self.landlord % 2:
                self.points += self.getPoints(self.current_trick.sum(axis=0))
            self.lead_player    = winner
            self.current_player = winner
            self.current_trick  = np.zeros((4, 54), dtype=int)
            if self.hands.sum() == 0:
                if winner % 2 != self.landlord % 2:
                    self.points += 2 * self.getPoints(self.kitty)
                self.done = True

        reward = self.get_reward(player)
        obs    = self.getObs(self.current_player)
        info   = {'landlord': self.landlord, 'landlord_team': self.landlord % 2}
        done   = self.done
        return obs, reward, done, info

    def get_reward(self, playerNum):
        if not self.done:
            return 0
        ranks = np.ceil(self.points / 20).astype(int)
        if self.points == 0:
            ranks = 0
        return -ranks if (playerNum % 2 == self.landlord % 2) else ranks

    def getObs(self, playerNum):
        parts = [
            self.hands[playerNum],
            self.played_cards[playerNum],
            self.played_cards[(playerNum+1)%4],
            self.played_cards[(playerNum+2)%4],
            self.played_cards[(playerNum+3)%4],
            self.current_trick[playerNum],
            self.current_trick[(playerNum+1)%4],
            self.current_trick[(playerNum+2)%4],
            self.current_trick[(playerNum+3)%4],
            np.array([self.points]),
            np.array([self.lead_player]),
            np.array([self.current_player]),
            np.array([self.lead_suit]),
            np.array([self.landlord]),
        ]
        return np.concatenate([p.flatten() for p in parts])

    def getPoints(self, totalcards):
        return (
            5*(totalcards[2] + totalcards[14] + totalcards[26] + totalcards[38]) +
            10*(totalcards[7] + totalcards[19] + totalcards[31] + totalcards[43] +
                 totalcards[10] + totalcards[22] + totalcards[34] + totalcards[46])
        )

    def compareCards(self, card1, card2):
        if card1 == card2:
            return 0
        if card1 >= 36 and card2 >= 36:
            if card1 in (48,49,50) and card2 in (48,49,50):
                return 0
            return 1 if card1 > card2 else -1
        if card1 >= 36:
            return 1
        if card2 >= 36:
            return -1
        if card1 // 12 == self.lead_suit and card2 // 12 == self.lead_suit:
            return 1 if card1 > card2 else -1
        if card1 // 12 == self.lead_suit:
            return 1
        if card2 // 12 == self.lead_suit:
            return -1
        return 0
