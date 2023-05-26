import numpy as np

class DataSampler:

    def __init__(self, data, output_info, log_frequency):
        self.data = data

        def is_categorical_column(column_info):
            return (len(column_info) == 1
                    and column_info[0].activation_fn == 'softmax')

        n_categorical_columns = sum(
            [1 for column_info in output_info if is_categorical_column(column_info)])

        self.categorical_column_matrix_st = np.zeros(
            n_categorical_columns, dtype='int32')

        # Store the row id for each category in each categorical column.
        # For example _rid_by_cat_cols[a][b] is a list of all rows with the
        # a-th categorical column equal value b.
        self.rid_by_cat_cols = []

        # Compute _rid_by_cat_cols
        st = 0
        for column_info in output_info:
            if is_categorical_column(column_info):
                span_info = column_info[0]
                ed = st + span_info.dim

                rid_by_cat = []
                for j in range(span_info.dim):
                    rid_by_cat.append(np.nonzero(data[:, st + j])[0])
                self.rid_by_cat_cols.append(rid_by_cat)
                st = ed
            else:
                st += sum([span_info.dim for span_info in column_info])
        assert st == data.shape[1]

        # Prepare an interval matrix for efficiently sample conditional vector
        max_category = max([
            column_info[0].dim
            for column_info in output_info
            if is_categorical_column(column_info)
        ], default=0)

        self.categorical_column_cond_st = np.zeros(n_categorical_columns, dtype='int32')
        self.categorical_column_n_category = np.zeros(n_categorical_columns, dtype='int32')
        self.categorical_column_category_prob = np.zeros((n_categorical_columns, max_category))
        self.n_categorical_columns = n_categorical_columns
        self.n_categories = sum([
            column_info[0].dim
            for column_info in output_info
            if is_categorical_column(column_info)
        ])

        st = 0
        current_id = 0
        current_cond_st = 0
        for column_info in output_info:
            if is_categorical_column(column_info):
                span_info = column_info[0]
                ed = st + span_info.dim
                category_freq = np.sum(data[:, st:ed], axis=0)
                if log_frequency:
                    category_freq = np.log(category_freq + 1)
                category_prob = category_freq / np.sum(category_freq)
                self.categorical_column_category_prob[current_id, :span_info.dim] = category_prob
                self.categorical_column_cond_st[current_id] = current_cond_st
                self.categorical_column_n_category[current_id] = span_info.dim
                current_cond_st += span_info.dim
                current_id += 1
                st = ed
            else:
                st += sum([span_info.dim for span_info in column_info])

    def random_choice_prob_index(self, categorical_column_id):
        probs = self.categorical_column_category_prob[categorical_column_id]
        r = np.expand_dims(np.random.rand(probs.shape[0]), axis=1)
        return (probs.cumsum(axis=1) > r).argmax(axis=1)

    def sample_condvec(self, batch):
        """Generate the conditional vector for training.

        Returns:
            cond (batch x #categories):
                The conditional vector.
            mask (batch x #categorical columns):
                A one-hot vector indicating the selected categorical column.
            categorical column id (batch):
                Integer representation of mask.
            category_id_in_col (batch):
                Selected category in the selected categorical column.
        """
        if self.n_categorical_columns == 0:
            return None

        categorical_column_id = np.random.choice(
            np.arange(self.n_categorical_columns), batch)

        cond = np.zeros((batch, self.n_categories), dtype='float32')
        mask = np.zeros((batch, self.n_categorical_columns), dtype='float32')
        mask[np.arange(batch), categorical_column_id] = 1
        category_id_in_col = self.random_choice_prob_index(categorical_column_id)
        category_id = (self.categorical_column_cond_st[categorical_column_id] + category_id_in_col)
        cond[np.arange(batch), category_id] = 1

        return cond, mask, categorical_column_id, category_id_in_col

    def sample_original_condvec(self, batch):
        """Generate the conditional vector for generation use original frequency."""
        if self.n_categorical_columns == 0:
            return None

        cond = np.zeros((batch, self.n_categories), dtype='float32')

        for i in range(batch):
            row_idx = np.random.randint(0, len(self.data))
            col_idx = np.random.randint(0, self.n_categorical_columns)
            matrix_st = self.categorical_column_matrix_st[col_idx]
            matrix_ed = matrix_st + self.categorical_column_n_category[col_idx]
            pick = np.argmax(self.data[row_idx, matrix_st:matrix_ed])
            cond[i, pick + self.categorical_column_cond_st[col_idx]] = 1

        return cond

    def sample_data(self, n, col, opt):
        """Sample data from original training data satisfying the sampled conditional vector.

        Returns:
            n rows of matrix data.
        """
        if col is None:
            idx = np.random.randint(len(self.data), size=n)
            return self.data[idx]

        idx = []
        for c, o in zip(col, opt):
            idx.append(np.random.choice(self.rid_by_cat_cols[c][o]))

        return self.data[idx]

    def dim_cond_vec(self):
        """Return the total number of categories."""
        return self.n_categories

    def generate_cond_from_condition_column_info(self, condition_info, batch):
        """Generate the condition vector."""
        vec = np.zeros((batch, self.n_categories), dtype='float32')
        id_ = self.categorical_column_matrix_st[condition_info['categorical_column_id']]
        id_ += condition_info['value_id']
        vec[:, id_] = 1
        return vec