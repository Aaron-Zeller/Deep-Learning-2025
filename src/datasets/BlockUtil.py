def run_addition(a: str, b: str) -> list[str]:
    """
    Addition Block Steps

                12345
    1 carry  -> ?
    2 a      ->   294
    3 b      -> + 123
    4 result -> =

    1. Column Rule: Completion when corresponding result entry is filled
    2. Column Steps:
       - Initial Step: No carry in next column, no result (+1 only once - final state after each column)
       - Carry Step:   Carry either 0 or 1 in next column, no result
       - Result Step:  Carry either 0 or 1 in next column, result in result entry is computed
       - Final Step:   Copy the carry down in case it is 1
    3. Completion:   Replace ? by $ in top left position of addition block.

               0       0      10      10     010     010    $010
      294     294     294     294     294     294     294     294
    + 123   + 123   + 123   + 123   + 123   + 123   + 123   + 123
    =       =       =   7   =   7   =  17   =  17   = 417   = 417

    Number of Steps = ________________
    """

    # Internal Length Variables
    max_digits = max(len(a), len(b))
    len_a = len(a)
    len_b = len(b)
    deficit_a = -min(0, len_a - len_b)
    deficit_b = -min(0, len_b - len_a)

    # Static Line Utilities
    carry_line = lambda x: "?" + "".join(str(d) if d >= 0 else " " for d in x[::-1])
    a_line = lambda x: " " * (2 + deficit_a) + x
    b_line = lambda x: "+" + " " * (deficit_b + 1) + x
    result_line = lambda x: "=" + "".join(str(d) if d >= 0 else " " for d in x[::-1])
    make_step = lambda _carry, _a, _b, _result: "\n".join([carry_line(_carry), a_line(_a), b_line(_b), result_line(_result)])

    # Algorithm Steps
    carry = [-1 for _ in range(max_digits + 1)]
    result = [-1 for _ in range(max_digits + 1)]

    # Initial Step
    steps = [make_step(carry, a, b, result)]

    for i in range(max_digits):
        # First Column has no Carry
        c = 0 if i == 0 else carry[i]

        # Get Digits and Compute Sum
        da = int(a[-i - 1]) if (i < len_a) else 0
        db = int(b[-i - 1]) if (i < len_b) else 0
        s = da + db + c

        # Carry Step
        carry[i + 1] = s // 10
        steps.append(make_step(carry, a, b, result))

        # Result Step
        result[i] = s % 10
        steps.append(make_step(carry, a, b, result))

    # Final Step
    if carry[-1] == 1:
        result[-1] = carry[-1]
        steps.append(make_step(carry, a, b, result))

    # Completion Step
    comp_line = lambda x: "$" + "".join(str(d) if d >= 0 else " " for d in x[::-1])
    make_step_comp = lambda _carry, _a, _b, _result: "\n".join([comp_line(_carry), a_line(_a), b_line(_b), result_line(_result)])
    steps.append(make_step_comp(carry, a, b, result))

    return steps

# TODO: Make compatible for numbers of different length
def run_subtraction(a: str, b: str) -> list[str]:
    """
    Subtraction Block Steps
                12345
    1 swap   -> 
    2 borrow ->
    3 a      ->   456
    4 b      -> - 892
    5 result -> =

    1. Valid Setup: a >= b -> else do swap states (+ 3 * max_digits + 1), result row empty borrow row empty, swap row empty
    + 1 due to the - sign being added 

    swaps left to right direction

    1. Column Rule: Completion when corresponding result entry is filled
    2. Column Steps: (Once valid)
       - Initial Step: No borrow in next column, no result (+1 only once - final state after each column / or after swap)
       - Borrow Step:  Borrow either 0 or 1 in next column, no result
       - Result Step:  Borrow either 0 or 1 in next column, result in result entry is computed
    3. Completion:  Replace ? by $ in top left position of subtraction block.

    ============================================ SWAPS ============================================
   
    ?       ?       ? 4     ? 45    ? 456   ? 456   ? 456   ? 456   ?  56   ?   6   ?      ?
    
      456     456      56       6             8       89      892     892     892     892     892     
    - 892   - 892   - 892   - 892   - 892   -  92   -   2   -       - 4     - 45    - 456   - 456
    =       =       =       =       =       =       =       =       =       =       =       =-      

    ============================================ VALID ============================================
    ?       ?       ?       ?       ?      ?        ?       $
               1       1      01      01     001     001     001
      892     892     892     892     892     892     892     892    
    - 456   - 456   - 456   - 456   - 456   - 456   - 456   - 456  
    =-      =-      =-  6   =-  6   =- 36   =- 36   =-436   =-436

    Number of Steps = ________________
    """

    # Internal Length Variables
    len_a = len(a)
    len_b = len(b)
    max_digits = max(len_a, len_b)
    deficit_a = -min(0, len_a - len_b)
    deficit_b = -min(0, len_b - len_a)
    
    # Static Line Utilities
    swap_line = lambda x: "?" + " " * 1 + "".join(str(d) if d >= 0 else " " for d in x[::-1])
    borrow_line = lambda x: " " * 1 + "".join(str(d) if d >= 0 else " " for d in x[::-1])
    a_line = lambda x: " " * 2 + "".join(str(d) if d >= 0 else " " for d in x[::-1])
    b_line = lambda x: "-" + " " * 1 + "".join(str(d) if d >= 0 else " " for d in x[::-1])
    result_line = lambda x, _sign: "=" + ("-" if _sign else " ") +  "".join(str(d) if d >= 0 else " " for d in x[::-1])
    make_step = lambda _swap, _borrow, _a, _b, _out, _sign: "\n".join([swap_line(_swap), borrow_line(_borrow), a_line(_a), b_line(_b), result_line(_out, _sign)])

    # Algorithm Steps
    swap = [-1 for _ in range(max_digits + 1)]
    borrow = [-1 for _ in range(max_digits + 1)]
    result = [-1 for _ in range(max_digits)]
    a_state = []
    a_state.extend(reversed([int(d) for d in a]))
    a_state.extend([-1 for _ in range(deficit_a)])
    b_state = []
    b_state.extend(reversed([int(d) for d in b]))
    b_state.extend([-1 for _ in range(deficit_b)])
    sign = False

    # Initial Step
    steps = [make_step(swap, borrow, a_state, b_state, result, sign)]

    # Swap Steps
    if(int(b) > int(a)):

        # Also Swap A and B to Ensure Correct Result
        temp = a
        a = b
        b = temp

        # Also Change Length Variables
        len_a = len(a)
        len_b = len(b)

        # Set Sign Before to Ensure that After Switching Setup is Correct
        sign = True
        steps.append(make_step(swap, borrow, a_state, b_state, result, sign))

        for stage in range(3):
            for i in range(max_digits):
                if stage == 0:
                    # Copy from A to Swap Line
                    swap[i] = a_state[i] # Copy to Swap Line
                    steps.append(make_step(swap, borrow, a_state, b_state, result, sign))

                    # Remove from A Line
                    a_state[i] = -1
                    steps.append(make_step(swap, borrow, a_state, b_state, result, sign))

                elif stage == 1:
                    # Copy from B to A Line
                    a_state[i] = b_state[i]
                    steps.append(make_step(swap, borrow, a_state, b_state, result, sign))

                    # Remove from B Line
                    b_state[i] = -1
                    steps.append(make_step(swap, borrow, a_state, b_state, result, sign))
                else:
                    # Copy from Swap Line to B
                    b_state[i] = swap[i]
                    steps.append(make_step(swap, borrow, a_state, b_state, result, sign))

                    # Remove from Swap Line
                    swap[i] = -1
                    steps.append(make_step(swap, borrow, a_state, b_state, result, sign))
    
    # Computation Steps
    for i in range(max_digits):
        # First Line has no Borrow
        borDebt = 0 if i == 0 else borrow[i]

        # Determine Borrow Value
        da = 0 if i >= len_a else int(a[-i - 1])
        db = 0 if i >= len_b else int(b[-i - 1])
        borLoan = 1 if (da < db) else 0

        # Compute Result
        diff = (da + borLoan * 10) - (db + borDebt)

        # Carry Step
        borrow[i + 1] = borLoan
        steps.append(make_step(swap, borrow, a_state, b_state, result, sign))

        # Result Step
        result[i] = diff
        steps.append(make_step(swap, borrow, a_state, b_state, result, sign))

    # Completion Step
    comp_line = lambda x: "$" + " " + "".join(str(d) if d >= 0 else " " for d in x[::-1])
    make_step_comp = lambda _swap, _borrow, _a, _b, _out, _sign: "\n".join([comp_line(_swap), borrow_line(_borrow), a_line(_a), b_line(_b), result_line(_out, _sign)])
    steps.append(make_step_comp(swap, borrow, a_state, b_state, result, sign))
    return steps

def run_pipe(a: str, b: str, u: str, d: str) -> list[str]:
    """
    Pipe Block Steps
                             12345

    1          sender         7364
    ...
    u          progress  
    u + 1      pipe          ~u  b
    ...
    u + b + 1  receiver 

    progress made from left to right
    only valid if len(a) == len(b)

    1 <= u, d <= 9


    1. Column Rule: Completion when corresponding when sender entry is placed at receiver entry
    2. Column Steps: 
       - Initial Step:  Progress is empty (+1)
       - Move Step:     Entry is copied to receiver entry
       - Progress Step: Progress is tracked with . in progress entry.
    3. Completion:      Replace ? by $ in first position of progress row
    
     7364    7364    7364    7364    7364    7364    7364    7364    7364    7364    7364     
    ?       ?       ?.      ?.      ?..     ?..     ?...    ?...    ?....   ?....   $....
    ~1  1   ~1  1   ~1  1   ~1  1   ~1  1   ~1  1   ~1  1   ~1  1   ~1  1   ~1  1   ~1  1   
             7       7       73      73      736     736     7364    7364    7364    7364

    Number of Steps = ________________
    """
    max_digits = max(len(a), len(b))

    # Static Line Utilities
    progress_line = lambda x: "?" + "".join(str(d) if d != "-1" else " " for d in x[::-1]) + "\n"
    pipe_line = lambda: "~" + str(u) + ((max_digits - 2) * " ") + str(d) + "\n"
    a_line = lambda x: " " + "".join(str(d) if d >= 0 else " " for d in x[::-1]) + "\n"
    b_line = lambda x: " " + "".join(str(d) if d >= 0 else " " for d in x[::-1]) + "\n"
    make_step = lambda _progress, _a, _b: "".join([a_line(_a), "" if u == 1 else "\n" * (int(u) - 1), progress_line(_progress), pipe_line(), "" if d == 1 else "\n" * (int(d) - 1), b_line(_b)])

    # Algorithm Steps
    a_state = []
    a_state.extend(reversed([int(d) for d in a]))
    b_state = []
    b_state.extend(reversed([int(d) for d in b]))
    progress_state = ["-1" for _ in range(max_digits)]

    # Initial Step
    steps = [make_step(progress_state, a_state, b_state)]

    for i in range(max_digits):
        # Copy Digit down
        b_state[i] = a_state[i]
        steps.append(make_step(progress_state, a_state, b_state))

        # Track Progress
        progress_state[i] = '.'
        steps.append(make_step(progress_state, a_state, b_state))    

    # Completion Step
    compl_line = lambda x: "$" + "".join(str(d) if d != "-1" else " " for d in x[::-1]) + "\n"   
    make_step_comp = lambda _progress, _a, _b: "".join([a_line(_a), "" if u == 1 else "\n" * (int(u) - 1), compl_line(_progress), pipe_line(), "" if d == 1 else "\n" * (int(d) - 1), b_line(_b)])
    steps.append(make_step_comp(progress_state, a_state, b_state))    

    return steps

def run_accumulation(a: str, b: str, max_length: int = None):
    """
    Accumulation Block Steps
                    12345
    1 progress   ->
    2 carry      ->
    3 a          ->    294
    4 b / result -> += 123

    1. Column Rule: Completion when corresponding result entry is filled and progress is tracked
    2. Column Steps:
       - Initial Step: No carry in next column, no result (+1 only once - final state after each column)
       - Carry Step:    Carry either 0 or 1 in next column, no result
       - Result Step:   Carry either 0 or 1 in next column, result in result entry is computed
       - Progress Step: Track Progress by using .
       - Final Step:    Copy the carry down in case it is 1
    3. Completion:  Replace ? by $ in top left position
    ?        ?        ?    .   ?    .   ?   ..   ?   ..   ?  ...   $  ...
                 0        0       10       10      010      010      010
       294      294      294      294      294      294      294      294
    += 123   += 123   += 127   += 127   += 117   += 117   += 417   += 417  

    Number of Steps = ________________
    """

    # If max_length is None then we Assume Single Addition
    if max_length is None:
        max_length = max(len(a), len(b))

    # Internal Length Variables
    max_digits = max(len(a), len(b))
    len_a = len(a)
    len_b = len(b)
    deficit_a = -min(0, len_a - len_b)
    deficit_b = -min(0, len_b - len_a)
    max_diff = max_length - max_digits

    # Static Line Utilities
    progress_line = lambda x: "?" + " " * (2 + max_diff) + "".join(str(d) if d != "-1" else " " for d in x[::-1]) 
    carry_line = lambda x: " " * (2 + max_diff) + "".join(str(d) if d >= 0 else " " for d in x[::-1])
    a_line = lambda x: " " * (3 + deficit_a + max_diff) + x
    b_line = lambda x: "+" + "=" + " " * (max_diff + 1) + "".join(str(d) if d >= 0 else " " for d in x[::-1]) 
    make_step = lambda _progress, _carry , _a, _b: "\n".join([progress_line(_progress), carry_line(_carry), a_line(_a), b_line(_b)])

    # Algorithm Steps
    carry = [-1 for _ in range(max_digits + 1)]
    b_state = []
    b_state.extend(reversed([int(d) for d in b]))
    b_state.extend([0 for _ in range(deficit_b)])
    print("deficit_b:", deficit_b)
    print(b_state)
    progress_state = ["-1" for _ in range(max_digits)]

    # Initial Step
    steps = [make_step(progress_state, carry, a, b_state)]

    for i in range(max_digits):
        # First Column has no Carry
        c = 0 if i == 0 else carry[i]

        # Get Digits and Compute Sum
        da = int(a[-i - 1]) if (i < len_a) else 0
        db = int(b[-i - 1]) if (i < len_b) else 0
        s = da + db + c

        # Carry Step
        carry[i + 1] = s // 10
        steps.append(make_step(progress_state, carry, a, b_state))

        # Decrease Deficit of b in Case of Extension
        #if i >= len_b:
        #    deficit_b = max(0, deficit_b - 1)

        # Result Step
        b_state[i] = s % 10
        steps.append(make_step(progress_state, carry, a, b_state))

        # Progress Step
        progress_state[i] = "."
        steps.append(make_step(progress_state, carry, a, b_state))

    # Final Step
    if carry[-1] == 1:
        b_state[-1] = carry[-1]
        steps.append(make_step(progress_state, carry, a, b_state))

    # Completion Step
    comp_line = lambda x: "$" + " " * (2 + max_diff) + "".join(str(d) if d != "-1" else " " for d in x[::-1]) 
    make_step_comp = lambda _progress, _carry, _a, _b: "\n".join([comp_line(_progress), carry_line(_carry), a_line(_a), b_line(_b)])
    steps.append(make_step_comp(progress_state, carry, a, b_state))

    return steps

def run_multiplication(a: str, b: str, max_length = None):
    """
    Multiplication Block Steps

    1 -> position on first number           (Position Row)
    2 -> progress on second number          (Progress Row)
    3 -> update row to ensure consistency   (Update Row)
    4 -> a
    5 -> b
    6 -> ==========================
    7 -> |   ACCUMULATION BLOCK   |
    8 -> ==========================

    - Progress is tracked in row 2: Once the dot . moves it does not go back
    - The dot in the first position always iterates through all positions
    - 1. Compute digit multiplication result
    - 2. Setup accumulation with ?
    - 3. Go to next position
    - 4. Run accumulation until $ appears

    1. Position Rule: Completion when position has progressed and accumulation has been completed
    2. Position Steps:
       - Initial Step:          Accumulation block has $ in top left entry position is marked via the two dots in position rows
       - Digit Steps:           Compute the result of the digit multiplication and add the corresponding numbers of zeroes
       - Setup Step:            Add ? to top left of accumulation block + add digit multiplication result to accumulator summand row
       - Position Step:         Go to next position:
                                - If lower dot (progress row) has reached left end ((len(b) - 1) steps) and 
                                  upper dot (position row) has reached left end we are done -> mark as complete + go to accumulation step
                                - Else if upper dot (position row) has reached left end ((len(a) - 1) steps):
                                - Copy lower dot down into update row in same column
                                - Delete lower dot from progress row
                                - Move upper dot (position row) to right top corner: copy to left then remove old position
                                - Copy lower dot from update row into progress row shifted one to the left
                                - Remove lower dot from update row
                                - Else: Move upper dot (position row) one to the left: copy to left then remove old position
        - Accumulation Steps:   Run accumulation block until $ appears
    3. Completion: Mark upper top left position with $ and Accumulator is also marked with $
    """

    # If max_length is None then we Assume Single Addition
    if max_length is None:
        max_length = len(a) + len(b) + 2

    # Internal Length Variables
    len_a = len(a)
    len_b = len(b)
    prod_len = len_a + len_b
    max_digits = prod_len + 2
    max_len = max(len_a, len_b)
    max_diff = max_length - max_digits
    deficit_a = -min(0, len_a - len_b)
    deficit_b = -min(0, len_b - len_a)

    # Static Line Utilities
    position_line = lambda x: "?" + " " * (max_digits - len_a + max_diff) + "".join(str(d) if d != "-1" else " " for d in x[::-1])
    position_line_comp = lambda x: "$" + " " * (max_digits - len_a + max_diff) + "".join(str(d) if d != "-1" else " " for d in x[::-1])
    progress_line = lambda x: " " * (max_digits - len_b + max_diff + 1) + "".join(str(d) if d != "-1" else " " for d in x[::-1])
    update_row =  lambda x: " " * (max_digits - len_b + max_diff + 1) + "".join(str(d) if d != "-1" else " " for d in x[::-1])
    a_line = lambda x: " " * (max_digits - len_a + max_diff + 1) + x
    b_line = lambda x: "*" + " " * (max_digits - len_b + max_diff) + x
    accum_lines = lambda x: x
    make_step = lambda _pos, _prog, _update, _a, _b, _accum: "\n".join([position_line(_pos), progress_line(_prog), update_row(_update), a_line(_a), b_line(_b), accum_lines(_accum)])
    make_step_comp = lambda _pos, _prog, _update, _a, _b, _accum: "\n".join([position_line_comp(_pos), progress_line(_prog), update_row(_update), a_line(_a), b_line(_b), accum_lines(_accum)])

    # Algorithm Steps
    position = ["-1" for _ in range(len_a)]
    position[0] = "."
    progress = ["-1" for _ in range(len_b)]
    progress[0] = "."
    update = ["-1" for _ in range(len_b)]
    accumulated_result = 0

    # Internal State Variables
    accum_progress = ["-1" for _ in range(prod_len)]
    accum_carry = [-1 for _ in range(prod_len)]
    accum_summand = [-1 for _ in range(prod_len)]
    accum_result = [-1 for _ in range(prod_len)]
    
    # Join Helper Function
    ap_str = lambda x, state: state + " " * (2 + max_diff) + "".join(str(d) if d != "-1" else " " for d in x[::-1])
    ac_str = lambda x: " " * (2 + max_diff) + "".join(str(d) if d >= 0 else " " for d in x[::-1]) + " "
    as_str = lambda x: " " * (3 + max_diff) + "".join(str(d) if d >= 0 else " " for d in x[::-1])
    ar_str = lambda x: "+=" + " " * (1 + max_diff) + "".join(str(d) if d >= 0 else " " for d in x[::-1])
    accum_block = lambda _prog, _state, _carry, _sum, _res: "\n".join([ap_str(_prog, _state), ac_str(_carry), as_str(_sum), ar_str(_res)])
    accum_block_as_strings = lambda _prog, _carry, _sum, _res: "\n".join([_prog, _carry, _sum, _res])

    # Initial Step
    steps = [make_step(position, progress, update, a, b, accum_block(accum_progress, "$", accum_carry, accum_summand, accum_result))]

    # Setup Accumulator
    accum_result[0] = 0
    steps.append(make_step(position, progress, update, a, b, accum_block(accum_progress, "$" , accum_carry, accum_summand, accum_result)))

    # Progress Steps - column by column
    for prog in range(len_b): 
        # Position Steps - go through all positions to finish current column
        for pos in range(len_a):
            # Get digit product
            da = 0 if pos >= len_a else int(a[-pos - 1])
            db = 0 if prog >= len_b else int(b[-prog - 1])
            dprod = da * db

            # Compute Indices
            ones_idx = prog + pos
            tens_idx = ones_idx + 1

            # Setup accumulation with ?
            steps.append(make_step(position, progress, update, a, b, accum_block(accum_progress, "?", accum_carry, accum_summand, accum_result)))

            # Reset the Summand Completely
            for i in range(prod_len):
                if accum_summand[i] != -1:
                    accum_summand[i] = -1
                    steps.append(make_step(position, progress, update, a, b, accum_block(accum_progress, "?", accum_carry, accum_summand, accum_result)))

            # Reset the Progress Completely
            for i in range(prod_len):
                if accum_progress[i] != "-1":
                    accum_progress[i] = "-1"
                    steps.append(make_step(position, progress, update, a, b, accum_block(accum_progress, "?", accum_carry, accum_summand, accum_result)))
            
            # Reset the Carry Completely
            for i in range(prod_len):
                if accum_carry[i] != -1:
                    accum_carry[i] = -1
                    steps.append(make_step(position, progress, update, a, b, accum_block(accum_progress, "?", accum_carry, accum_summand, accum_result)))

            # Add the First Digit - if it exists
            if dprod >= 10:
                accum_summand[tens_idx] = dprod // 10
                steps.append(make_step(position, progress, update, a, b, accum_block(accum_progress, "?", accum_carry, accum_summand, accum_result)))
            
            # Add the Second Digit
            accum_summand[ones_idx] = dprod % 10
            steps.append(make_step(position, progress, update, a, b, accum_block(accum_progress, "?", accum_carry, accum_summand, accum_result)))

            # Add the Remaining Zeroes
            for i in range(ones_idx):
                accum_summand[i] = 0
                steps.append(make_step(position, progress, update, a, b, accum_block(accum_progress, "?", accum_carry, accum_summand, accum_result)))

            # Zero-Pad if Summand is Larger than Result
            if accum_result[tens_idx] != -1:
                accum_result[i] = 0
                steps.append(make_step(position, progress, update, a, b, accum_block(accum_progress, "?", accum_carry, accum_summand, accum_result)))

            # Run the Accumulator
            dprod = dprod * (10**(prog + pos))
            steps_accum = run_accumulation(str(dprod), str(accumulated_result), prod_len + max_diff)
        
            # Save the Result Internally for Easy Handling
            accumulated_result = accumulated_result + dprod
            
            # Use Different Functions to Differentiate Between Completion and Continuation
            func = make_step

            # Update Position According to Rules:
            #              COMPLETION
            if prog == len_b - 1 and pos == len_a - 1:
                func = make_step_comp

            elif pos == len_a - 1:
                # Copy Lower Dot Down into Update Row in Same Column
                update[prog] = "."
                steps.append(make_step(position, progress, update, a, b, accum_block(accum_progress, "?", accum_carry, accum_summand, accum_result)))

                # Delete Lower Dot from Progress Row
                progress[prog] = "-1"
                steps.append(make_step(position, progress, update, a, b, accum_block(accum_progress, "?", accum_carry, accum_summand, accum_result)))

                # Move Upper Dot to Right Top Corner - Copy Step 
                if pos != 0: # Ensures: No Redundant Step
                    position[0] = "."
                    steps.append(make_step(position, progress, update, a, b, accum_block(accum_progress, "?", accum_carry, accum_summand, accum_result)))

                # Move Upper Dot to Right Top Corner - Deletion Step 
                if pos != 0: # Ensures: No Redundant Step
                    position[pos] = "-1"
                    steps.append(make_step(position, progress, update, a, b, accum_block(accum_progress, "?", accum_carry, accum_summand, accum_result)))

                # Copy Lower Dot from Update Row into Progress Row Shifted One to the Left
                progress[prog + 1] = "."
                steps.append(make_step(position, progress, update, a, b, accum_block(accum_progress, "?", accum_carry, accum_summand, accum_result)))

                # Remove Lower Dot from Update Row
                update[prog] = "-1"
                steps.append(make_step(position, progress, update, a, b, accum_block(accum_progress, "?", accum_carry, accum_summand, accum_result)))
                      
            else:
                # Move Upper Dot One to the Left - Copy Step
                position[pos + 1] = "."
                steps.append(make_step(position, progress, update, a, b, accum_block(accum_progress, "?", accum_carry, accum_summand, accum_result)))
            
                # Move Upper Dot One to the Left - Deletion Step
                position[pos] = "-1"
                steps.append(make_step(position, progress, update, a, b, accum_block(accum_progress, "?", accum_carry, accum_summand, accum_result)))

            # Accumulation Step 
            for step in steps_accum:
                lines = step.splitlines()
                    
                # Make Steps
                accum_progress = ["." for c in reversed(lines[0]) if c == "."]
                accum_carry = [int(c) for c in reversed(lines[1]) if c.isdigit()]
                accum_summand = [int(c) for c in reversed(lines[2]) if c.isdigit()]
                accum_result = [int(c) for c in reversed(lines[3]) if c.isdigit()]

                # Keep Length Constant
                accum_progress = accum_progress + ["-1" for _ in range(prod_len - len(accum_progress))] 
                accum_carry = accum_carry + [-1 for _ in range(prod_len - len(accum_carry))] 
                accum_summand = accum_summand + [-1 for _ in range(prod_len - len(accum_summand))] 
                accum_result = accum_result + [-1 for _ in range(prod_len - len(accum_result))] 
                steps.append(func(position, progress, update, a, b, accum_block_as_strings(lines[0], lines[1], lines[2], lines[3])))    

    return steps


def run_decrementation(a: str, b: str, max_length: int):
    """

    PRE: It must hold that b >= a. (NO SWAPPING)

    1. COMPUTE
    2. CLEAR A ROW
    3. SHIFT B TO A ROW

    Decrementation Block Steps
                    123456
    1 progress   ->
    2 borrow     ->
    3 a / result ->    456
    4 b          -> -= 892

    1. Column Rule: Completion when corresponding result entry is filled
    2. Column Steps: (Once valid)
       - Initial Step:  No borrow in next column, no result (+1 only once - final state after each column / or after swap)
       - Borrow Step:   Borrow either 0 or 1 in next column, no result
       - Result Step:   Borrow either 0 or 1 in next column, result in result entry is computed
       - Progress Step: Track Progress in Progress Row in Current Column
    3. Completion:  Replace ? by $ in top left position of decrementation block 

    Number of Steps = ________________
    """

    # If max_length is None then we Assume Single Addition
    if max_length is None:
        max_length = max(len(a), len(b))

    # Internal Length Variables
    len_a = len(a)
    len_b = len(b)
    max_digits = max(len_a, len_b)
    deficit_a = -min(0, len_a - len_b)
    deficit_b = -min(0, len_b - len_a)
    max_diff = max_length - max_digits
    
    # Static Line Utilities
    progress_line = lambda x: "?" + " " * (1 + max_diff) + "".join(str(d) if d != "-1" else " " for d in x[::-1])
    borrow_line = lambda x: " " * (1 + max_diff) + "".join(str(d) if d >= 0 else " " for d in x[::-1])
    a_line = lambda x: " " * (2 + max_diff) + "".join(str(d) if d >= 0 else " " for d in x[::-1])
    b_line = lambda x: "-=" + max_diff * " " + "".join(str(d) if d >= 0 else " " for d in x[::-1])
    make_step = lambda _progress, _borrow, _a, _b: "\n".join([progress_line(_progress), borrow_line(_borrow), a_line(_a), b_line(_b)])

    # Algorithm Steps
    progress = ["-1" for _ in range(max_digits)]
    borrow = [-1 for _ in range(max_digits + 1)]
    a_state = []
    a_state.extend(reversed([int(d) for d in a]))
    a_state.extend([-1 for _ in range(deficit_a)])
    b_state = []
    b_state.extend(reversed([int(d) for d in b]))
    b_state.extend([-1 for _ in range(deficit_b)])

    # Initial Step
    steps = [make_step(progress, borrow, a_state, b_state)]
    
    # Computation Steps
    for i in range(max_digits):
        # First Line has no Borrow
        borDebt = 0 if i == 0 else borrow[i]

        # Determine Borrow Value
        da = 0 if i >= len_a else int(a[-i - 1])
        db = 0 if i >= len_b else int(b[-i - 1])
        borLoan = 1 if (da < db) else 0

        # Compute Result
        diff = (da + borLoan * 10) - (db + borDebt)

        # Carry Step
        borrow[i + 1] = borLoan
        steps.append(make_step(progress, borrow, a_state, b_state))

        # Result Step
        b_state[i] = diff
        steps.append(make_step(progress, borrow, a_state, b_state))

        # Progress Step
        progress[i] = "."
        steps.append(make_step(progress, borrow, a_state, b_state))


    # Completion Step
    comp_line = lambda x: "$" + " " + "".join(str(d) if d != "-1" else " " for d in x[::-1])
    make_step_comp = lambda _progress, _borrow, _a, _b: "\n".join([comp_line(_progress), borrow_line(_borrow), a_line(_a), b_line(_b)])
    steps.append(make_step_comp(progress, borrow, a_state, b_state))

    # Clear Steps 
    for i in range(max_digits):
        a_state[i] = -1
        steps.append(make_step_comp(progress, borrow, a_state, b_state))

    # Shift Steps - Copy Digits from a to b Row
    for i in range(max_digits):
        a_state[i] = b_state[i]
        steps.append(make_step_comp(progress, borrow, a_state, b_state))

    # Shift Steps - Remove Digits in b Row
    for i in range(max_digits):
        b_state[i] = -1
        steps.append(make_step_comp(progress, borrow, a_state, b_state))

    return steps

def run_division(a: str, b: str):
    """

    Only valid if a >= b

    1  -> Progress Row
    2  -> a
    3  -> b
    4  -> ==================
    5  -> |                |
    6  -> |                |
    7  -> | MULTIPLICATION |
    8  -> |      BLOCK     |
    9  -> |                |
    10 -> |                |
    11 -> ==================
    12 -> | 1 PIPE BLOCK 4 |
    13 -> ==================
    14 -> | DECREMENTATION |
    15 -> |     BLOCK      |
    16 -> ==================
    17 -> Result Row

    Initial Setup: All blocks have a $ and have their initial setup

    1. Add a to a row in decrementation block

    2. Add b to a row in multiplication row

    3. For i = 1, ... , len_a:
        I.       Compute largest single digit multiple (1-9) of 10^i * b that is still smaller than the value in the decrementation a / result row (this can be skipped) -> jump to step X immediately
        II.      Add digit to multiplication block b row at column len_a - i - 1 
        III.     Reset progress and set $ to ? in multiplication block
        IV.      Run multiplication block
        V.       Reset progress and set $ to ? in pipe block
        VI.      Run pipe block
        VII.     Reset progress and set $ to ? in subtraction block
        VIII.    Run subtraction block
        IX.      Shift result digit down (Copy and Delete from multiplication block)
        X.       Track Progress 

    4. Completion: All blocks have $ in them -> Set ? to $ in division block
    
    """

    # Internal Length Variables
    len_a = len(b)
    len_b = len(a)
    max_digits = len_a
    diff_len = len_a - len_b

    # Static Line Utilities
    progress_line = lambda x: "?" + " " * 2 + "".join(str(d) if d != "-1" else " " for d in x[::-1])
    progress_line_comp = lambda x: "$" + " " * 2 + "".join(str(d) if d != "-1" else " " for d in x[::-1])
    a_line = lambda x: " " * 3 + "".join(str(d) if d != "-1" else " " for d in x[::-1])
    b_line = lambda x: "/" + " " * (2 + diff_len) + "".join(str(d) if d != "-1" else " " for d in x[::-1])
    result_line = lambda x: "=" + " " * 2 + "".join(str(d) if d != "-1" else " " for d in x[::-1])
    make_step = lambda _prog, _a, _b, _res, _mult, _pipe, _decum: "\n".join([progress_line(_prog), a_line(_a), b_line(_b), _mult, _pipe, _decum, result_line(_res)])
    make_step_comp = lambda _prog, _a, _b, _res, _mult, _pipe, _decum: "\n".join([progress_line_comp(_prog), a_line(_a), b_line(_b), _mult, _pipe, _decum, result_line(_res)])

    # Algorithm Steps
    progress = ["-1" for _ in range(max_digits)]
    result = [-1 for _ in range(max_digits)]
    remainder = 0

    """
    =============== MULTIPLICATION BLOCK ===============
    """

    # Internal State Variables - Multiplication Block
    mult_position = ["-1" for _ in range(max_digits)]
    mult_progress = ["-1" for _ in range(max_digits)]
    mult_b = [-1 for _ in range(max_digits)]
    mult_result = [-1 for _ in range(max_digits)]

    # Join Helper Function - Multiplication Block
    mpos_str = lambda x, state: state + " " * 1 + "".join(str(d) if d != "-1" else " " for d in x[::-1])
    mprog_str = lambda x: " " * 2 + "".join(str(d) if d != "-1" else " " for d in x[::-1])
    ma_str = lambda: " " * (2 + max_digits - len_b) + b
    mb_str = lambda x: " " * (2 + max_digits - len(x)) + x 
    maccum_prog = lambda x: " " * 2 + "".join(str(d) if d != "-1" else " " for d in x[::-1])
    maccum_carry = lambda x: " " * 2 + "".join(str(d) if d != "-1" else " " for d in x[::-1])
    maccum_sum = lambda x: " " * 2 + "".join(str(d) if d != "-1" else " " for d in x[::-1])
    # mult_block = lambda _pos, _prog, _mb, 
  
    # This is just old stugg
    ap_str = lambda x, state: state + " " * 2 + "".join(str(d) if d != "-1" else " " for d in x[::-1])
    ac_str = lambda x: 2 * " " + "".join(str(d) if d >= 0 else " " for d in x[::-1]) + " "
    as_str = lambda x: 3 * " " + "".join(str(d) if d >= 0 else " " for d in x[::-1])
    ar_str = lambda x: "+= " + "".join(str(d) if d >= 0 else " " for d in x[::-1])
    accum_block = lambda _prog, _state, _carry, _sum, _res: "\n".join([ap_str(_prog, _state), ac_str(_carry), as_str(_sum), ar_str(_res)])
    accum_block_complete = lambda _prog, _carry, _sum, _res: "\n".join([_prog, _carry, _sum, _res])

    """
    =============== DECREMENTATION BLOCK ===============
    """

    # Internal State Variables - Decrementation Block


    



steps = run_multiplication("37", "19")
for step in steps:
    print(step)
    print("==========")

