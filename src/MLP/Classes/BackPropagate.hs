{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE QuantifiedConstraints #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE UndecidableInstances #-}
module MLP.Classes.BackPropagate (BackPropagateCPU (..)) where

import Control.Arrow (second)
import Control.Category

import Prelude hiding ((.), id)

import MLP.Classes.FeedForward
import MLP.Core

class FeedForwardCPU l => BackPropagateCPU l where
  backPropagateCPU :: Double -> l i o -> (TensorTree i, TensorTree o) -> ((TensorTree o, TensorTree i), l i o)

newtype BPCPU c i o = BPCPU {runBPCPU :: (TensorTree i, TensorTree o) -> ((TensorTree o, TensorTree i), Network c i o)}

instance Category (BPCPU c) where
  id = BPCPU $ \(i, o) -> ((o, i), id)
  BPCPU f . BPCPU g = BPCPU $ \(ia, og) ->
    let ((ma, ig), g') = g (ia, mg)
        ((oa, mg), f') = f (ma, og)
    in ((oa, ig), f' . g')

instance (forall l. c l => BackPropagateCPU l) => BackPropagateCPU (Network c) where
  backPropagateCPU learningRate = runBPCPU . free (\l -> BPCPU $ second forget . backPropagateCPU learningRate l)

