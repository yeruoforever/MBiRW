using DelimitedFiles
using Random
using Plots

function Laplacian_normalization(M)
    for i = 1:size(M, 1)
        M[i, i] = sum(M[i, :])
    end
    for i ∈ 1:size(M, 1)
        for j ∈ 1:size(M, 2)
            M[i, j] = M[i, j] / sqrt(M[i, i] * M[j, j])
        end
    end
    return M
end

function MBiRW(simR, simD, A, α, l, r)
    MR = Laplacian_normalization(simR)
    MD = Laplacian_normalization(simD)
    A = A / sum(A)
    RD = A
    for i ∈ 1:max(l, r)
        dflag = 1
        rflag = 1
        if i ≤ l
            Rr = α * RD * MR + (1 - α) * A
            rflag = 1
        end
        if i ≤ r
            Rd = α * MD * RD + (1 - α) * A
            dflag = 1
        end
        RD = (rflag * Rr + dflag * Rd) / (rflag + dflag)
    end
    return RD
end

function part_of_test_indexs(indexs, fold, f_id)
    h, w = size(indexs)
    group_size = fld(w, fold)
    if f_id == fold
        return @view indexs[:, (f_id-1)*group_size+1:end]
    else
        return @view indexs[:, (f_id-1)*group_size+1:f_id*group_size]
    end
end

function test_indexs(A::Matrix)
    h, w = size(A)
    m = vcat([shuffle(Vector(1:w))' for i ∈ 1:h]...)
end

function train_matrix(A::Matrix, indexs)
    m = copy(A)
    for i ∈ 1:size(A, 1)
        for j ∈ indexs[i, :]
            m[i, j] = 0
        end
    end
    m
end

function evalution_vector(outputs, sorted_outputs, labels, threshold)
    pair = zip(outputs, labels)
    tp = sum(pair) do (pre, target)
        target == 1 && pre >= threshold
    end
    fn = sum(pair) do (pre, target)
        target == 1 && pre < threshold
    end
    fp = sum(pair) do (pre, target)
        target == 0 && pre >= threshold
    end
    tn = sum(pair) do (pre, target)
        target == 0 && pre < threshold
    end
    (TP = tp, FN = fn, FP = fp, TN = tn)
end


function fold_cross(A::Matrix,α,l,r,fold = 10)
    t_indexs = test_indexs(A)
    for i ∈ 1:fold
        indexs = part_of_test_indexs(t_indexs, fold, i)
        ma = train_matrix(A, indexs)
        rd = MBiRW(simR, simD, ma, α, l, r)
        pre = [rd[k, j] for k ∈ 1:size(indexs, 1) for j ∈ indexs[k, :]]
        thresholds = sort(pre)
        labels = [A[k, j] for k ∈ 1:size(indexs, 1) for j ∈ indexs[k, :]]
        p1 = plot(zeros(0))
        p2 = plot(zeros(0))
        for threshold in thresholds
            em = evalution_vector(pre, thresholds, labels, threshold)
            R = em.TP / (em.TP + em.FN)
            P = em.TP / (em.TP + em.FP)
            TPR = R
            FPR = em.FP / (em.TN + em.FP)
            push!(p1, (FPR, TPR))
            push!(p2, (R, P))
        end
        display(plot(p1, p2, layout = (1, 2), legend = false))
    end
end


simR = readdlm("./Datasets/DrugSimMat")
simD = readdlm("./Datasets/DiseaseSimMat")
A = readdlm("./Datasets/DiDrAMat")
α = 0.3
l = 3
r = 3
fold_cross(A,α,l,r)
