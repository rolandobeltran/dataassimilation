# Data Assimilation with Python
Let me begin with a very simple example: suppose you drop a ball off a cliff, and you want to estimate the ball's velocity using Newton's Laws of Motion. 
Now, please imagine that, in a moment, a little breeze hits the ball, and thus, the trajectory that you calculated initially with the Newton Formula will not capture the real value of the velocity.
But don't be afraid, also, you have an electronic hand device that can measure the velocity of the ball. 
Unfortunally the device is not perfect, so there is an error in the measures.
At this point you have two imperfect estimation of a variable value. The first made from a math model, and the other obtained from a sensor.


Data assimilation is the problem of combining two sources of information to produce an optimal estimation of a variable's value.
In the most common setup, we have, on the one hand, a math model and, on the other, a sensor.
