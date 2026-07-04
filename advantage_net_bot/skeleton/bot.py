'''
Parent class of all pokerbots.
'''


class Bot():
    '''
    Player API.
    '''
    def handle_new_round(self, game_state, round_state, active):
        pass

    def handle_round_over(self, game_state, terminal_state, active):
        pass

    def get_action(self, game_state, round_state, active):
        pass
