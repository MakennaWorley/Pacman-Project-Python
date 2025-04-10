# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        ghostPositions = [ghost.getPosition() for ghost in newGhostStates]

        #away from ghosts
        for ghostPos, scares in zip(ghostPositions, newScaredTimes):
            if manhattanDistance(newPos, ghostPos) < 2:
                return float("-inf")

        score = 0
        foodPos = newFood.asList()

        #eat food
        if currentGameState.getPacmanPosition() in foodPos:
            score += 10

        #go to food
        if foodPos:
            distance = min(manhattanDistance(newPos, food) for food in foodPos)
            score += 10.0 / distance

        return successorGameState.getScore() + score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        actions = gameState.getLegalActions(0)
        score = float("-inf")
        move = None

        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            value = self.minimax(successor, 0, 1)

            if value > score or move is None:
                score = value
                move = action

        return move

    def minimax(self, state, depth, agentIndex):
        if state.isWin() or state.isLose() or depth >= self.depth:
            return self.evaluationFunction(state)

        agentNext = (agentIndex + 1) % state.getNumAgents()
        if agentNext == 0:
            depth += 1

        actions = state.getLegalActions(agentIndex)
        if not actions:
            return self.evaluationFunction(state)

        values = [self.minimax(state.generateSuccessor(agentIndex, action), depth, agentNext) for action in actions]
        return max(values) if agentIndex == 0 else min(values)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        actions = gameState.getLegalActions(0)
        score, alpha, beta = float("-inf"), float("-inf"), float("inf")
        move = None

        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            value = self.alphaBeta(successor, 0, alpha, beta, 1)

            if value > score or move is None:
                score = value
                move = action
            alpha = max(alpha, value)

        return move

    def alphaBeta(self, state, depth, alpha, beta, agentIndex):
        if state.isWin() or state.isLose() or depth >= self.depth:
            return self.evaluationFunction(state)

        agentNext = (agentIndex + 1) % state.getNumAgents()
        if agentNext == 0:
            depth += 1

        actions = state.getLegalActions(agentIndex)
        if not actions:
            return self.evaluationFunction(state)

        if agentIndex == 0: #pacman
            value = float("-inf")

            for action in actions:
                value = max(value, self.alphaBeta(state.generateSuccessor(agentIndex, action), depth, alpha, beta, agentNext))

                if value > beta:
                    return value
                alpha = max(value, alpha)

            return value
        else: #ghosts
            value = float("inf")

            for action in actions:
                value = min(value, self.alphaBeta(state.generateSuccessor(agentIndex, action), depth, alpha, beta, agentNext))

                if value < alpha:
                    return value
                beta = min(value, beta)

            return value

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        actions = gameState.getLegalActions(0)
        score = float("-inf")
        move = None

        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            value = self.expectimax(successor, 0, 1)

            if value > score or move is None:
                score = value
                move = action

        return move

    def expectimax(self, state, depth, agentIndex):
        if state.isWin() or state.isLose() or depth >= self.depth:
            return self.evaluationFunction(state)

        agentNext = (agentIndex + 1) % state.getNumAgents()
        if agentNext == 0:
            depth += 1

        actions = state.getLegalActions(agentIndex)
        if not actions:
            return self.evaluationFunction(state)

        if agentIndex == 0:
            return max(self.expectimax(state.generateSuccessor(agentIndex, action), depth, agentNext) for action in actions)
        else:
            total = sum(self.expectimax(state.generateSuccessor(agentIndex, action), depth, agentNext) for action in actions)
            return total

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
    Pacman should try to eat the pellets close to together and not leave pellets alone or scattered.
    Pacman should try to run away from ghosts very quickly and not get close to the ghosts.
    Pacman should try to eat all the capsules quickly.
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    capsules = currentGameState.getCapsules()
    ghostPositions = [ghost.getPosition() for ghost in newGhostStates]

    if currentGameState.isWin():
        return float("inf")
    if currentGameState.isLose():
        return float("-inf")

    value = currentGameState.getScore()

    #eat a lot of food fast
    if newFood:
        foodDistList =[manhattanDistance(newPos, food) for food in newFood]
        foodMin = min(foodDistList)
        foodAvg = sum(foodDistList) / len(foodDistList)

        value += 15.0 / (foodMin + 1) #eat closest food
        value -= .25 * foodAvg #eat food close to eachother
        value -= 3 * len(foodDistList) #clear board of food

    #run away from ghosts
    for ghostPos, scareTime in zip(ghostPositions, newScaredTimes):
        dist = manhattanDistance(newPos, ghostPos)
        if scareTime == 0:
            if dist < 2:
                value -= 50
            else:
                value -= 50.0 / dist
        else:
            value += 40.0 / (dist + 1)

    #eat capsules
    value -= 5 * len(capsules)
    if capsules:
        minCapsule = min(manhattanDistance(newPos, cap) for cap in capsules)
        value += 6.0 / minCapsule

    return value

# Abbreviation
better = betterEvaluationFunction
