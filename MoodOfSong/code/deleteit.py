import re

#Replace all white-space characters with the digit "9":

str = "Where is the world going to look for the heat?" \
      " It doesn't fade. The feeling of your heartbeat is the most confused." \
      " The half-second is in your broad shoulders. Have the courage to step forward. Happiness never counts back and no calculation. A method to determine what is right or wrong, just know that you are not tired, you are disappointed with me, fear that the world is changing, the harder it is, the more you love, the more you love, so you are expecting flowers to fall, at least this season is covered by sweetness, and you will jump into love. Heavy and growing love, even if everything is not recorded, the weakness will be more angry with the situation on the way, even if there is some emotion, let me learn to hold on to the future, don’t let the hand let go, the most confused, half a second, have your broad shoulders. I have the courage to step forward, I am happy, I want to go to see the polar times, hundreds of people, no identity, what is the reason, I want to stop, I just need to know, don’t let you down, don’t be afraid of the world, change the harder, the more difficult you are, the more you love. In anticipation of flowering and falling, at least this season has been covered by sweetness. Nothing jumps into love. The sea is getting more and more love. Even if everything is not recorded, the weakness will be tempered and the situation will not be cut and you will be tied in the face. meaning The more difficult it is, the more difficult it is to love, the more it is willing to stand up and defend the love flower. At least this season has been covered by the sweetness. The harmlessness jumps into the sea, the love is heavy, the love wave has been covered, the tears will be covered, and the situation will be contested. I can't let go of my arms."
x = re.sub("\t", " ", str)
print(str)
print('  ')
print(x)
