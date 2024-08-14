## Lambda Functions

Anonymous functions are mapped to uniquely named global functions with a default rewrite of their body.

Constraints:

1. they should be uniquqe based on their types, arguments and body
2. Their name is a version of that
3. Their egg value is their name but simplified
4. Their pretty/serialized version (for graphviz) is that version but without the types.
