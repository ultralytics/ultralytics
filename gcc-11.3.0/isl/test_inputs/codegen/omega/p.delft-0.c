if (P2 >= 0 && P2 <= 3 && P1 == P2)
  for (int c0 = 0; c0 <= min(2, -P2 + 4); c0 += 1)
    for (int c2 = (-P2 - c0 + 6) % 3; c2 <= 3; c2 += 3)
      s0(c0, c0, c2, c2);
