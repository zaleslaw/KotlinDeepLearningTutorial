package tensorflow.old_api.graph

/**
 * Defines the simplest Operand Graph.
 */
fun main() {
    /*Graph().use { g ->
        Session(g).use { session ->
            val tf = Ops.create(g)
            val aOps = tf.constant(10L)
            val bOps = tf.constant(5L)

            Tensors.create(10L).use { c1 ->
                Tensors.create(5L).use { c2 ->
                    AutoCloseableList(
                        s.runner()
                            .feed(x1, c1)
                            .feed(x2, c2)
                            .fetch(grads0[0])
                            .fetch(grads1[0])
                            .fetch(grads1[1])
                            .run()
                    ).use { outputs ->
                        println(outputs.size)
                        println(outputs[0].floatValue())
                        println(outputs[1].floatValue())
                        println(outputs[2].floatValue())
                    }


            Output<Float> = g.opBuilder("Square", "y1")
                .addInput(x1)
                .build()
                .output<Float>(0)

            val input = arrayOf(aOps, bOps)
            val addOps: Output<Long> = g.opBuilder("Add", "Add").addInputList(input).build().output(0);

            println(session.runner().fetch(addOps).run()[0].longValue())
        }
    }*/
}