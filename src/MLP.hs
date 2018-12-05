{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE NamedFieldPuns #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE QuantifiedConstraints #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
module MLP (module MLP.Core, module MLP.Classes.FeedForward, module MLP.Classes.BackPropagate, module MLP) where

import Control.Applicative
import Control.Category
import Control.DeepSeq
import Control.Exception (evaluate)
import Control.Monad
import Control.Monad.Primitive
import Data.Foldable (toList)
import Data.List (foldl')
import GHC.TypeNats
import Linear
import Linear.V
import System.Random.MWC
import System.Random.MWC.Distributions

import Prelude hiding ((.), id)

import MLP.Classes.BackPropagate
import MLP.Classes.FeedForward
import MLP.Core

class RandomInitialize l where
  randomInitialize :: PrimMonad m => l i o -> Gen (PrimState m) -> m (l i o)

newtype RI m c i o = RI {runRI :: Gen (PrimState m) -> m (Network c i o)}

instance Applicative m => Category (RI m c) where
  id = RI $ \_ -> pure id
  RI f . RI g = RI $ (liftA2 . liftA2) (.) f g

instance (forall l. c l => RandomInitialize l) => RandomInitialize (Network c) where
  randomInitialize = runRI . free (RI . fmap (fmap forget) . randomInitialize)

class NormalFormNetwork l where
  normalFormNetwork :: l i o -> ()

data NFN i o = NFN

instance Category NFN where
  id = NFN
  NFN . NFN = NFN

instance (forall l. c l => NormalFormNetwork l) => NormalFormNetwork (Network c) where
  normalFormNetwork net = case free (\l -> normalFormNetwork l `seq` NFN) net of
    NFN -> ()

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

-- Add bias layers

data AddBias i o where
  AddBias :: { bias :: Double } -> AddBias ('Tip '[]) ('Tip '[])

instance FeedForwardCPU AddBias where
  feedForwardCPU AddBias {bias} a = a + bias

instance BackPropagateCPU AddBias where
  backPropagateCPU learningRate AddBias {bias} (ia, og) =
    ((ia + bias, og), AddBias {bias = bias + learningRate * og})

instance RandomInitialize AddBias where
  randomInitialize AddBias {} gen = AddBias <$> standard gen 

instance NormalFormNetwork AddBias where
  normalFormNetwork AddBias {bias} = rnf bias

-- Sigmoid layer

data Sigmoid i o where
  Sigmoid :: Sigmoid ('Tip '[]) ('Tip '[])

sigmoidFunction :: Double -> Double
sigmoidFunction x = 1/(1 + exp (negate x))

instance FeedForwardCPU Sigmoid where
  feedForwardCPU Sigmoid = sigmoidFunction

instance BackPropagateCPU Sigmoid where
  backPropagateCPU _ Sigmoid (ia, og) =
    let oa = sigmoidFunction ia
        ig = sigmoidFunction ia * (1 - sigmoidFunction ia) * og
    in ((oa, ig), Sigmoid)

instance RandomInitialize Sigmoid where
  randomInitialize Sigmoid _ = pure Sigmoid

instance NormalFormNetwork Sigmoid where
  normalFormNetwork Sigmoid = ()

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

-- training
--
-- List of (input, output) pairs
-- final cost function: quadratic difference
-- hooking it all together


train 
  :: (forall l. c l => (BackPropagateCPU l, NormalFormNetwork l)
     ,KnownNat o
     )
  => [(V i Double, V o Double)]
  -> Double
  -> Network c ('Tip '[i]) ('Tip '[o])
  -> Network c ('Tip '[i]) ('Tip '[o])
train dat learningRate n = foldl' trainExample' n dat
  where
    trainExample net (input, ref) = 
      let ((output, _), net') = backPropagateCPU learningRate net (input, cost)
          cost = output - ref
      in net'

    trainExample' net ir =
      let net' = trainExample net ir
      in normalFormNetwork net' `seq` net'
