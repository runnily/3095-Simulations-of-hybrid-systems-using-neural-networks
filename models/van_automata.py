"""
    Arthur: Adanna Obibuaku
    Purpose: The purpose of this is too build the van der pool model
"""
from automata import Automata
import decimal

class VanAutomata(Automata):

    def run(self, y0, x0, delta, num_simulations):

        decimal.getcontext().prec = 1000

        x_data = ''
        y_data = ''
        oscillate = ''

        time = 0
        y =  decimal.Decimal(y0)
        x =  decimal.Decimal(x0)
        
        for _ in range(num_simulations):
            dxdtime = decimal.Decimal(y)

            dydtime = decimal.Decimal(decimal.Decimal(0.3) * ( 1 - (x**2)) * (y-x))

            print('{time},{x},{y},{dxdtime},{dydtime},"{state}"\n'.format(time=time, x=x, y=y,dxdtime=dxdtime, dydtime=dydtime, state=self.current.name))
            x_data += '{time},{x},{dxdtime}\n'.format(time=time, x=x, dxdtime=dxdtime)
            y_data += '{time},{y},{dydtime}\n'.format(time=time, y=y, dydtime=dydtime)
            oscillate += '{time},{x},{y}\n'.format(time=time, x=x, y=y)

            x += dxdtime*decimal.Decimal(delta)
            y += dydtime*decimal.Decimal(delta)
            time += delta
            time = round(time,1)

        self.save(x_data, ['time', 'x', 'dxdtime'], 'data/oscillate/x_data.csv')
        self.save(y_data, ['time', 'y', 'dydtime'], 'data/oscillate/y_data.csv')
        self.save(oscillate, ['time', 'x', 'y'], 'data/oscillate/oscillate.csv')




