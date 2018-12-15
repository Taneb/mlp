{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE NamedFieldPuns #-}
module MLP.Layers.FullyConnected where

import Control.DeepSeq
import GHC.TypeNats
import Linear
import Linear.V
import System.Random.MWC.Distributions

import MLP.Core
import MLP.Classes

-- Fully connected layers

data FullyConnected i o where
  FullyConnected :: (KnownNat i, KnownNat o) => { weights :: V o (V i Double) } -> FullyConnected ('Tip '[i]) ('Tip '[o])

instance FeedForwardCPU FullyConnected where
  feedForwardCPU (FullyConnected {weights}) v = weights !* v

instance BackPropagateCPU FullyConnected where
  backPropagateCPU learningRate FullyConnected {weights} (ia, og) =
    let ig = transpose weights !* og
    in ((weights !* ia, ig), FullyConnected {weights = weights !+! learningRate *!! og `outer` ia})

instance RandomInitialize FullyConnected where
  randomInitialize FullyConnected {weights} gen = FullyConnected <$> traverse (traverse (\_ -> standard gen)) weights

instance NormalFormNetwork FullyConnected where
  normalFormNetwork FullyConnected {weights} = rnf weights

