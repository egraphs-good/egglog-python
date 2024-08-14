[
    Datatype(
        "Math",
        [
            Variant("Num", ["i64"], None),
            Variant("Var", ["String"], None),
            Variant("Add", ["Math", "Math"], None),
            Variant("Mul", ["Math", "Math"], None),
        ],
    ),
    ActionCommand(
        Let(
            Span(
                SrcFile(
                    "test.egg",
                    '(datatype Math\n          (Num i64)\n          (Var String)\n          (Add Math Math)\n          (Mul Math Math))\n\n        ;; expr1 = 2 * (x + 3)\n        (let expr1 (Mul (Num 2) (Add (Var "x") (Num 3))))',
                ),
                151,
                200,
            ),
            "expr1",
            Call(
                Span(
                    SrcFile(
                        "test.egg",
                        '(datatype Math\n          (Num i64)\n          (Var String)\n          (Add Math Math)\n          (Mul Math Math))\n\n        ;; expr1 = 2 * (x + 3)\n        (let expr1 (Mul (Num 2) (Add (Var "x") (Num 3))))',
                    ),
                    162,
                    199,
                ),
                "Mul",
                [
                    Call(
                        Span(
                            SrcFile(
                                "test.egg",
                                '(datatype Math\n          (Num i64)\n          (Var String)\n          (Add Math Math)\n          (Mul Math Math))\n\n        ;; expr1 = 2 * (x + 3)\n        (let expr1 (Mul (Num 2) (Add (Var "x") (Num 3))))',
                            ),
                            167,
                            174,
                        ),
                        "Num",
                        [
                            Lit(
                                Span(
                                    SrcFile(
                                        "test.egg",
                                        '(datatype Math\n          (Num i64)\n          (Var String)\n          (Add Math Math)\n          (Mul Math Math))\n\n        ;; expr1 = 2 * (x + 3)\n        (let expr1 (Mul (Num 2) (Add (Var "x") (Num 3))))',
                                    ),
                                    172,
                                    173,
                                ),
                                Int(2),
                            )
                        ],
                    ),
                    Call(
                        Span(
                            SrcFile(
                                "test.egg",
                                '(datatype Math\n          (Num i64)\n          (Var String)\n          (Add Math Math)\n          (Mul Math Math))\n\n        ;; expr1 = 2 * (x + 3)\n        (let expr1 (Mul (Num 2) (Add (Var "x") (Num 3))))',
                            ),
                            175,
                            198,
                        ),
                        "Add",
                        [
                            Call(
                                Span(
                                    SrcFile(
                                        "test.egg",
                                        '(datatype Math\n          (Num i64)\n          (Var String)\n          (Add Math Math)\n          (Mul Math Math))\n\n        ;; expr1 = 2 * (x + 3)\n        (let expr1 (Mul (Num 2) (Add (Var "x") (Num 3))))',
                                    ),
                                    180,
                                    189,
                                ),
                                "Var",
                                [
                                    Lit(
                                        Span(
                                            SrcFile(
                                                "test.egg",
                                                '(datatype Math\n          (Num i64)\n          (Var String)\n          (Add Math Math)\n          (Mul Math Math))\n\n        ;; expr1 = 2 * (x + 3)\n        (let expr1 (Mul (Num 2) (Add (Var "x") (Num 3))))',
                                            ),
                                            185,
                                            188,
                                        ),
                                        String("x"),
                                    )
                                ],
                            ),
                            Call(
                                Span(
                                    SrcFile(
                                        "test.egg",
                                        '(datatype Math\n          (Num i64)\n          (Var String)\n          (Add Math Math)\n          (Mul Math Math))\n\n        ;; expr1 = 2 * (x + 3)\n        (let expr1 (Mul (Num 2) (Add (Var "x") (Num 3))))',
                                    ),
                                    190,
                                    197,
                                ),
                                "Num",
                                [
                                    Lit(
                                        Span(
                                            SrcFile(
                                                "test.egg",
                                                '(datatype Math\n          (Num i64)\n          (Var String)\n          (Add Math Math)\n          (Mul Math Math))\n\n        ;; expr1 = 2 * (x + 3)\n        (let expr1 (Mul (Num 2) (Add (Var "x") (Num 3))))',
                                            ),
                                            195,
                                            196,
                                        ),
                                        Int(3),
                                    )
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        )
    ),
]