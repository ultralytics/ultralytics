if (M >= 3 && N >= 4)
  for (int c0 = 1; c0 < (2 * M + 2 * N - 2) / 5; c0 += 1)
    for (int c1 = max(c0 - (M + 2) / 5, (c0 + 1) / 2); c1 <= min(min(c0, (M + 2 * N) / 5 - 1), (2 * N + 5 * c0 + 1) / 10); c1 += 1)
      for (int c2 = max(max(max(max(0, c0 - c1 - 1), c1 - (N + 6) / 5 + 1), c0 - (M + N + 4) / 5 + 1), floord(-N + 5 * c0 - 3, 10) + 1); c2 <= min(min(min(c1, (M + N - 2) / 5), c0 - c1 + (N - 1) / 5 + 1), (N + 5 * c0 + 3) / 10); c2 += 1)
        for (int c3 = max(max(max(c0, 2 * c1 - (2 * N + 5) / 5 + 1), c1 + c2 - (N + 3) / 5), 2 * c2 - (N + 2) / 5); c3 <= min(min(min(min(min(c0 + 1, c1 + c2 + 1), c1 + (M - 2) / 5 + 1), 2 * c2 + (N - 2) / 5 + 1), (2 * M + 2 * N - 1) / 5 - 1), c2 + (M + N) / 5); c3 += 1)
          for (int c4 = max(max(max(max(c1, c0 - c2), c0 - (M + 6) / 5 + 1), c3 - (M + 2) / 5), (c3 + 1) / 2); c4 <= min(min(min(min(min(min(min(c0, c1 + 1), -c2 + c3 + (N - 1) / 5 + 1), c0 - c2 + N / 5 + 1), (M + 2 * N + 1) / 5 - 1), c2 + (N + 2) / 5), (2 * N + 5 * c0 + 3) / 10), (2 * N + 5 * c3 + 2) / 10); c4 += 1)
            S1(c0, c1, c2, c3, c4, c2);
