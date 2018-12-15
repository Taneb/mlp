{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE TypeOperators #-}
module MLP.Layers.Lift where

import Control.Applicative
import Control.DeepSeq
import Data.Foldable (toList)
import GHC.TypeNats
import Linear.V

import MLP.Core
import MLP.Classes

-- Lift layers to higher dimension tensors

data LiftLayer l i o where
  LiftLayer :: KnownNat x => V x (l ('Tip i) ('Tip o)) -> LiftLayer l ('Tip (x ': i)) ('Tip (x ': o))

instance FeedForwardCPU l => FeedForwardCPU (LiftLayer l) where
  feedForwardCPU (LiftLayer l) = liftA2 feedForwardCPU l 

instance BackPropagateCPU l => BackPropagateCPU (LiftLayer l) where
  backPropagateCPU lr (LiftLayer l) (ia, og) =
     let r = (curry . backPropagateCPU lr <$> l <*> ia <*> og)
         oa = fst.fst <$> r
         ig = snd.fst <$> r
         l' = snd     <$> r
     in ((oa, ig), LiftLayer l')

instance RandomInitialize l => RandomInitialize (LiftLayer l) where
  randomInitialize (LiftLayer l) gen = LiftLayer <$> traverse (\l' -> randomInitialize l' gen) l

instance NormalFormNetwork l => NormalFormNetwork (LiftLayer l) where
  normalFormNetwork (LiftLayer l) = liftRnf (normalFormNetwork) (toList l)

